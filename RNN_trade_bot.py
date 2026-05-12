import requests
import time
import json
import math
import sqlite3
import os
import numpy as np
import atexit
import csv
import torch
from threading import Lock, local, Event, Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import deque, defaultdict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import resource

# supposedly this prevents some crash I had 
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))

# --- CONFIGURATION ---
CATEGORY = "SPORTS"  
LEADERBOARD_SIZE = 1000
MAX_WORKERS = 25

DB_FILE = 'trades.db' 

DATA_URL = "https://data-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com"

MODEL_PATH = "rnn_history_model.pt"
MODEL_EDGE_SCALE = 10.0
MIN_EV_THRESHOLD = 0.05
MIN_BUY_PRICE = 0.05
MAX_BUY_PRICE = 0.96

# total weight needed to enter a watch 
WATCH_THRESHOLD = 0.1

RATE_LIMITS = {
    'data_general': 950, 
    'data_positions': 950,  
    'data_leaderboard': 950, 
    'clob_market': 1350,
    'gamma_markets': 300,  
}

LAST_LOOP_TIMINGS = {}
LAST_PROCESS_TIMINGS = {}

def filter_title(title):
    # returns True if title is a spread or VS match for league, dota, or CS
    if not title: return False
    return ('vs.' in title and (('?' not in title and ':' not in title) or 'O/U' in title)) or \
           any(x in title for x in ['Spread:', 'LoL', 'Dota 2', 'Counter-Strike'])

class RateLimiter:
    # ping APIs at just below rate limit
    def __init__(self):
        self.limits = RATE_LIMITS 
        self.window = 10.0
        self.history = defaultdict(deque)
        self.lock = Lock()

    def wait_if_needed(self, endpoint):
        limit = self.limits.get(endpoint, 100)
        while True:
            with self.lock:
                now = time.time()
                window_start = now - self.window
                while self.history[endpoint] and self.history[endpoint][0] <= window_start:
                    self.history[endpoint].popleft()
                if len(self.history[endpoint]) < limit:
                    self.history[endpoint].append(now)
                    return
                sleep_time = self.history[endpoint][0] + self.window - now
            if sleep_time > 0: time.sleep(sleep_time)

limiter = RateLimiter()

class TradesDB:
    def __init__(self, db_file, reset_trades=False, reset_sharps=False):
        self.db_file = db_file
        self.lock = Lock()
        self._thread_local = local()
        self._connections = []
        self._write_queue = Queue()
        self._writer_thread = Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        self._init_db(reset_trades, reset_sharps)
        atexit.register(self._cleanup)

    def _get_conn(self):
        """Get or create a thread-local database connection with proper configuration."""
        if not hasattr(self._thread_local, "conn"):
            conn = self._open_configured_conn()
            self._thread_local.conn = conn
            self._connections.append(conn)
        return self._thread_local.conn

    def _open_configured_conn(self):
        """Open a SQLite connection with robust WAL setup and safe fallback."""
        conn = sqlite3.connect(self.db_file, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.commit()
            return conn
        except sqlite3.OperationalError:
            conn.close()
            for suffix in ("-wal", "-shm"):
                try:
                    path = f"{self.db_file}{suffix}"
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            conn = sqlite3.connect(self.db_file, check_same_thread=False, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.commit()
            return conn

    def _column_exists(self, column_name):
        """Check if a column exists in the trades table."""
        try:
            conn = self._get_conn()
            cur = conn.execute("PRAGMA table_info(trades)")
            cols = [row[1] for row in cur.fetchall()]
            return column_name in cols
        except Exception:
            return False

    def _writer_loop(self):
        """Single-threaded writer loop to serialize all DB writes."""
        conn = self._open_configured_conn()

        while True:
            item = self._write_queue.get()
            if item is None:
                break

            fn, done = item
            try:
                result = fn(conn)
                conn.commit()
                done['result'] = result
            except Exception as e:
                conn.rollback()
                done['error'] = e
            finally:
                done['event'].set()

        try:
            conn.close()
        except Exception:
            pass

    def _write_sync(self, fn):
        """Run a write task on the writer thread and wait for completion."""
        done = {'event': Event()}
        self._write_queue.put((fn, done))
        done['event'].wait()
        if 'error' in done:
            raise done['error']
        return done.get('result')

    def _cleanup(self):
        """Close all database connections on shutdown."""
        try:
            self._write_queue.put(None)
            self._writer_thread.join(timeout=5)
        except Exception:
            pass
        for conn in self._connections:
            try:
                conn.close()
            except Exception:
                pass

    def _init_db(self, reset_trades, reset_sharps):
        def _init(conn):
            
            # always reset markets table on startup
            print("--- Resetting Markets Table ---")
            conn.execute("DROP TABLE IF EXISTS markets")
            
            # only reset PNL history if reset_trades is True
            if reset_trades:
                print("--- Resetting PNL History Table ---")
                conn.execute("DROP TABLE IF EXISTS pnl_history")
            
            if reset_sharps:
                print("--- Resetting Sharps Table ---")
                conn.execute("DROP TABLE IF EXISTS sharps")

            conn.execute('''
                CREATE TABLE IF NOT EXISTS sharps (
                    wallet TEXT PRIMARY KEY,
                    username TEXT,
                    sharpness_score REAL,
                    num_open INTEGER,
                    open_positions TEXT,
                    last_checked REAL
                )
            ''')

            if reset_trades:
                print("--- Resetting Trade Table ---")
                conn.execute("DROP TABLE IF EXISTS trades")

            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    condition_id TEXT,
                    token_id TEXT,
                    market_title TEXT,
                    outcome_idx INTEGER,
                    outcome_label TEXT,
                    status TEXT,
                    buy_time REAL,
                    buy_price REAL,
                    buy_edge REAL,
                    current_edge REAL,
                    buy_weights_for REAL,
                    buy_weights_against REAL,
                    shares_bought INTEGER,
                    current_price REAL,
                    pnl_per_share REAL,
                    pnl REAL,
                    history TEXT,
                    current_ev REAL
                )
            ''')
            
            # create indexes for better query performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_condition_status ON trades(condition_id, status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sharps_score ON sharps(sharpness_score)')

            # allow up to 2 open trades per condition_id (one per outcome)
            # this prevents duplicate trades on the same outcome while the market is still open
            conn.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_open_trade 
                ON trades(condition_id, outcome_idx) 
                WHERE status = 'OPEN'
            ''')
            
            # markets table for tracking spread statistics and slugs
            conn.execute('''
                CREATE TABLE IF NOT EXISTS markets (
                    condition_id TEXT PRIMARY KEY,
                    market_title TEXT,
                    slug TEXT,
                    token0 TEXT,
                    token1 TEXT,
                    token0_spread REAL,
                    token1_spread REAL,
                    num_checks INTEGER DEFAULT 0,
                    is_closed INTEGER DEFAULT 0,
                    last_checked REAL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_markets_condition ON markets(condition_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_markets_slug ON markets(slug)')
            
            # PNL tracking table for graphing over time
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pnl_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    total_investment REAL,
                    total_pnl REAL,
                    realized_pnl REAL,
                    unrealized_pnl REAL,
                    avg_pnl_per_share REAL,
                    roi REAL,
                    num_open_trades INTEGER,
                    num_closed_trades INTEGER,
                    avg_edge REAL,
                    total_shares INTEGER
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_pnl_timestamp ON pnl_history(timestamp)')
            return True

        for attempt in range(5):
            try:
                self._write_sync(_init)
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < 4:
                    print('problem writing to DB')
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise

    # --- Sharps Methods ---
    def get(self, wallet):
        conn = self._get_conn()
        cur = conn.execute("SELECT * FROM sharps WHERE wallet = ?", (wallet,))
        row = cur.fetchone()
        if row:
            data = dict(row)
            data['open_positions'] = json.loads(data['open_positions'])
            return data
        return None
    
    def get_all_cached_users(self):
        conn = self._get_conn()
        cur = conn.execute("SELECT wallet, username, sharpness_score, num_open, open_positions FROM sharps WHERE sharpness_score > 0.1")
        return [{'proxyWallet': r['wallet'], 'userName': r['username'], 'sharpness_score': r['sharpness_score'], 'num_open': r['num_open'], 'open_positions': r['open_positions']} for r in cur.fetchall()]

    # --- Markets Methods ---
    def update_market_spread(self, condition_id, token0, token1, token0_spread, token1_spread, market_title=None, slug=None):
        """Update market spread statistics with exponential moving average."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                def _write(conn):
                    # get existing data
                    cur = conn.execute(
                        "SELECT token0_spread, token1_spread, num_checks, slug FROM markets WHERE condition_id = ?",
                        (condition_id,)
                    )
                    row = cur.fetchone()

                    if row:
                        # update with moving average (alpha = 0.2)
                        alpha = 0.2
                        new_token0_spread = alpha * token0_spread + (1 - alpha) * row['token0_spread']
                        new_token1_spread = alpha * token1_spread + (1 - alpha) * row['token1_spread']
                        new_num_checks = row['num_checks'] + 1

                        conn.execute('''
                            UPDATE markets 
                            SET token0_spread = ?, token1_spread = ?, num_checks = ?, 
                                market_title = COALESCE(?, market_title),
                                slug = COALESCE(?, slug),
                                last_checked = ?
                            WHERE condition_id = ?
                        ''', (new_token0_spread, new_token1_spread, new_num_checks, market_title, slug, time.time(), condition_id))
                    else:
                        # insert new market
                        conn.execute('''
                            INSERT INTO markets (condition_id, market_title, slug, token0, token1, token0_spread, token1_spread, num_checks, last_checked)
                            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                        ''', (condition_id, market_title, slug, token0, token1, token0_spread, token1_spread, time.time()))
                    return True

                self._write_sync(_write)
                return
            except sqlite3.OperationalError as e:
                if 'locked' in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    raise

    def get_market_spread_stats(self, condition_id):
        """Get spread statistics for a market."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT token0_spread, token1_spread, num_checks, is_closed FROM markets WHERE condition_id = ?",
            (condition_id,)
        )
        row = cur.fetchone()
        if row:
            return {
                'token0_spread': row['token0_spread'],
                'token1_spread': row['token1_spread'],
                'num_checks': row['num_checks'],
                'is_closed': row['is_closed']
            }
        return None

    def mark_market_closed(self, condition_id):
        """Mark a market as closed."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                def _write(conn):
                    conn.execute('''
                        UPDATE markets 
                        SET is_closed = 1
                        WHERE condition_id = ?
                    ''', (condition_id,))
                    return True

                self._write_sync(_write)
                return
            except sqlite3.OperationalError as e:
                if 'locked' in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    raise

    def get_markets_to_check(self):
        """Get all markets with OPEN/WATCH trades that need to be checked for closure."""
        conn = self._get_conn()
        cur = conn.execute('''
            SELECT DISTINCT m.condition_id, m.slug 
            FROM markets m
            INNER JOIN trades t ON m.condition_id = t.condition_id
            WHERE m.is_closed = 0 
            AND m.slug IS NOT NULL 
            AND t.status IN ('OPEN', 'WATCH')
        ''')
        return [dict(row) for row in cur.fetchall()]

    def should_trade_market(self, condition_id, current_spread):
        """Check if market meets spread criteria for trading."""
        stats = self.get_market_spread_stats(condition_id)
        
        # check if market is closed
        if stats and stats.get('is_closed'):
            return False
        
        # current spread must be tight
        if current_spread > 0.03:
            return False
        
        # require at least 3 checks to ensure consistent liquidity
        if stats and stats['num_checks'] >= 3:
            avg_spread = (stats['token0_spread'] + stats['token1_spread']) / 2
            if avg_spread > 0.05:
                return False
        else:
            return False
        
        return True

    # --- PNL History Methods ---
    def record_pnl_snapshot(self, stats):
        """Record a PNL snapshot for historical tracking."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                def _write(conn):
                    conn.execute('''
                        INSERT INTO pnl_history 
                        (timestamp, total_investment, total_pnl, realized_pnl, unrealized_pnl, 
                         avg_pnl_per_share, roi, num_open_trades, num_closed_trades, avg_edge, total_shares)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        time.time(),
                        stats.get('total_investment', 0),
                        stats.get('total_pnl', 0),
                        stats.get('realized_pnl', 0),
                        stats.get('unrealized_pnl', 0),
                        stats.get('avg_pnl_per_share', 0),
                        stats.get('roi', 0),
                        stats.get('num_open', 0),
                        stats.get('num_closed', 0),
                        stats.get('avg_edge', 0),
                        stats.get('total_shares', 0)
                    ))
                    return True

                self._write_sync(_write)
                return
            except sqlite3.OperationalError as e:
                if 'locked' in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    raise

    def get_pnl_history(self, limit=None):
        """Get PNL history for graphing."""
        conn = self._get_conn()
        if limit:
            cur = conn.execute(
                "SELECT * FROM pnl_history ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        else:
            cur = conn.execute("SELECT * FROM pnl_history ORDER BY timestamp ASC")
        return [dict(row) for row in cur.fetchall()]

    def export_pnl_to_csv(self, filename='pnl_history.csv'):
        """Export PNL history to CSV file."""
        history = self.get_pnl_history()
        
        if not history:
            print("No PNL history to export")
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        
        print(f"Exported {len(history)} PNL records to {filename}")


    def update_sharp(self, wallet, data):
        """Update sharp data using INSERT ... ON CONFLICT for atomic upsert."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                def _write(conn):
                    conn.execute('''
                        INSERT INTO sharps 
                        (wallet, username, sharpness_score, num_open, open_positions, last_checked)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(wallet) DO UPDATE SET
                            username = COALESCE(excluded.username, sharps.username),
                            sharpness_score = excluded.sharpness_score,
                            num_open = excluded.num_open,
                            open_positions = excluded.open_positions,
                            last_checked = excluded.last_checked
                    ''', (
                        wallet, 
                        data.get('username'), 
                        data.get('sharpness_score'), 
                        data.get('num_open'),
                        json.dumps(data.get('open_positions', [])),
                        time.time()
                    ))
                    return True

                self._write_sync(_write)
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                print(f"Error updating sharp {wallet}: {e}")
                raise

    # --- Trade Methods ---
    def get_active_trades(self):
        """Get all OPEN/WATCH trades grouped by condition_id."""
        conn = self._get_conn()
        cur = conn.execute('''
            SELECT id, condition_id, outcome_idx, token_id, outcome_label, buy_price, buy_edge, shares_bought, market_title, status, history, current_ev
            FROM trades WHERE status IN ('OPEN', 'WATCH')
        ''')
        trades = cur.fetchall()
        grouped = defaultdict(list)
        for t in trades:
            grouped[t['condition_id']].append(dict(t))
        return grouped
    
    def get_all_monitored_trades(self):
        """Get all monitored trades (OPEN/WATCH status)."""
        conn = self._get_conn()
        cur = conn.execute('''
            SELECT id, condition_id, outcome_idx, token_id, outcome_label, buy_price, buy_edge, shares_bought, market_title, status
            FROM trades WHERE status IN ('OPEN', 'WATCH')
        ''')
        return [dict(t) for t in cur.fetchall()]

    def open_trade(self, trade_data):
        """
        Open a new trade position.
        Only allows one trade per (condition_id, outcome_label) pair.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                def _write(conn):
                    shares = trade_data['shares_bought']

                    # buy_edge is passed in and represents edge frozen at time of buying
                    buy_edge = round(trade_data['buy_edge'], 2)

                    # Store the weights for reference
                    weights_for = round(float(trade_data.get('sharps_for', 0)), 2)
                    weights_against = round(float(trade_data.get('sharps_against', 0)), 2)

                    print(f"$$$ BUYING: {trade_data['market_title']} | Edge: {buy_edge:.2f} | Shares: {shares} | Price: {trade_data['buy_price']:.4f}")

                    conn.execute('''
                        INSERT INTO trades 
                        (condition_id, token_id, market_title, outcome_idx, outcome_label, status, 
                         buy_time, buy_price, buy_edge, current_edge, buy_weights_for, buy_weights_against, shares_bought, current_price, history, current_ev)
                        VALUES (?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                        trade_data['condition_id'], 
                        trade_data.get('token_id'),
                        trade_data['market_title'], 
                        trade_data['outcome_idx'], 
                        trade_data.get('outcome_label', 'Unknown'),
                        time.time(),
                        trade_data['buy_price'], 
                        buy_edge,  # buy_edge frozen at time of buying
                        buy_edge,  # current_edge initialized to buy_edge
                        weights_for,
                        weights_against,
                        trade_data['shares_bought'],
                        trade_data['buy_price'],
                        None,
                        trade_data.get('current_ev')
                    ))
                    return True

                self._write_sync(_write)
                return
                    
            except sqlite3.IntegrityError as e:
                # this is expected when trying to open a duplicate trade
                print(f"Cannot open duplicate trade for {trade_data['condition_id']} - {trade_data.get('outcome_label', 'Unknown')}: {e}")
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                print(f"Error opening trade for {trade_data['condition_id']}: {e}")
                raise

    def open_watch_trade(self, trade_data):
        """
        Open a WATCH trade anchor (no real position) to track edge decay/increase.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                def _write(conn):
                    # avoid duplicate WATCH anchors for the same condition/outcome
                    cur = conn.execute('''
                        SELECT id FROM trades
                        WHERE condition_id = ? AND outcome_idx = ? AND status = 'WATCH'
                    ''', (trade_data['condition_id'], trade_data['outcome_idx']))
                    if cur.fetchone():
                        return True

                    buy_edge = round(trade_data['buy_edge'], 2)
                    weights_for = round(float(trade_data.get('sharps_for', 0)), 2)
                    weights_against = round(float(trade_data.get('sharps_against', 0)), 2)

                    print(f"👀 WATCHING: {trade_data['market_title']} | Edge: {buy_edge:.2f}")

                    conn.execute('''
                        INSERT INTO trades
                        (condition_id, token_id, market_title, outcome_idx, outcome_label, status,
                         buy_time, buy_price, buy_edge, current_edge, buy_weights_for, buy_weights_against, shares_bought, current_price, history, current_ev)
                        VALUES (?, ?, ?, ?, ?, 'WATCH', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                        trade_data['condition_id'],
                        trade_data.get('token_id'),
                        trade_data['market_title'],
                        trade_data['outcome_idx'],
                        trade_data.get('outcome_label', 'Unknown'),
                        time.time(),
                        trade_data.get('buy_price', 0),
                        buy_edge,
                        buy_edge,
                        weights_for,
                        weights_against,
                        0,
                        trade_data.get('buy_price', 0),
                        json.dumps([]),
                        trade_data.get('current_ev')
                    ))
                    return True

                self._write_sync(_write)
                return

            except sqlite3.IntegrityError as e:
                print(f"Cannot open duplicate WATCH for {trade_data['condition_id']} - {trade_data.get('outcome_label', 'Unknown')}: {e}")
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                print(f"Error opening WATCH for {trade_data['condition_id']}: {e}")
                raise

    def finalize_trade(self, trade_id, final_price, buy_price, shares_bought):
        """Finalize a trade with CLOSED status when market has resolved."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                def _write(conn):
                    pnl_per_share = round(final_price - buy_price, 2)
                    total_pnl = round(pnl_per_share * shares_bought, 2)

                    print(f"🔒 FINALIZING Trade #{trade_id} | Final Price: {final_price:.2f} | PnL: {total_pnl:.4f} | PnL/share: {pnl_per_share:.4f}")
                    conn.execute('''
                        UPDATE trades 
                        SET status = 'CLOSED', 
                            pnl_per_share = ?,
                            pnl = ?,
                            current_price = ?
                        WHERE id = ?
                    ''', (pnl_per_share, total_pnl, final_price, trade_id))
                    return True

                self._write_sync(_write)
                return
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                print(f"Error finalizing trade {trade_id}: {e}")
                raise

    def update_trade_prices(
        self,
        condition_id,
        outcome_idx,
        bid_price,
        ask_price,
        current_edge=None,
        weights_for=None,
        weights_against=None,
        n_sharps_for=None,
        n_sharps_against=None,
        avg_sharp_for=None,
        avg_sharp_against=None,
        avg_price_for=None,
        avg_price_against=None,
        current_ev=None,
        esport=None,
    ):
        """
        Continuously update current prices and current edge for OPEN/WATCH trades.
        """

        max_retries = 3
        for attempt in range(max_retries):
            try:
                def _write(conn):
                    # fetch OPEN/WATCH trades for this market/outcome
                    cur = conn.execute('''
                        SELECT id, buy_price, shares_bought, status, history
                        FROM trades
                        WHERE condition_id = ? AND outcome_idx = ? AND status IN ('OPEN', 'WATCH')
                    ''', (condition_id, outcome_idx))

                    trades = cur.fetchall()

                    for trade in trades:
                        current_price = bid_price
                        # for OPEN trades: calculate PnL based on current price
                        pnl_per_share = round(current_price - trade['buy_price'], 2)
                        total_pnl = round(pnl_per_share * (trade['shares_bought'] or 1), 2)

                        if trade['status'] == 'WATCH' and current_edge is not None and weights_for is not None and weights_against is not None:
                            history = []
                            if trade['history']:
                                try:
                                    history = json.loads(trade['history'])
                                except Exception:
                                    history = []
                            history.append([
                                bid_price,
                                ask_price,
                                current_edge,
                                weights_for,
                                weights_against,
                                time.time(),
                                n_sharps_for,
                                n_sharps_against,
                                avg_sharp_for,
                                avg_sharp_against,
                                avg_price_for,
                                avg_price_against,
                                current_ev,
                                esport
                            ])

                            conn.execute('''
                                UPDATE trades
                                SET current_price = ?,
                                    pnl_per_share = ?,
                                    pnl = ?,
                                    current_edge = ?,
                                    history = ?,
                                    current_ev = COALESCE(?, current_ev)
                                WHERE id = ?
                            ''', (current_price, pnl_per_share, total_pnl, current_edge, json.dumps(history), current_ev, trade['id']))
                        elif current_edge is not None:
                            conn.execute('''
                                UPDATE trades
                                SET current_price = ?,
                                    pnl_per_share = ?,
                                    pnl = ?,
                                    current_edge = ?,
                                    current_ev = COALESCE(?, current_ev)
                                WHERE id = ?
                            ''', (current_price, pnl_per_share, total_pnl, current_edge, current_ev, trade['id']))
                        else:
                            conn.execute('''
                                UPDATE trades 
                                SET current_price = ?, 
                                    pnl_per_share = ?,
                                    pnl = ?,
                                    current_ev = COALESCE(?, current_ev)
                                WHERE id = ?
                            ''', (current_price, pnl_per_share, total_pnl, current_ev, trade['id']))
                    return True

                self._write_sync(_write)
                return
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                print(f"Error updating prices for {condition_id}: {e}")
                raise

    def scale_trade(self, trade_id, new_shares, new_avg_price, new_edge):
        """Scale up an existing trade position."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                def _write(conn):
                    print(f"+++ SCALING Trade #{trade_id} | New Shares: {new_shares} | Current Edge: {new_edge:.2f}")
                    conn.execute('''
                        UPDATE trades 
                        SET shares_bought = ?, 
                            buy_price = ?
                        WHERE id = ?
                    ''', (new_shares, new_avg_price, trade_id))
                    return True

                self._write_sync(_write)
                return
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                print(f"Error scaling trade {trade_id}: {e}")
                raise

# --- GAMMA API FUNCTIONS ---
def fetch_markets_by_slug(session, slug):
    """Fetch market data from GAMMA API by slug."""
    try:
        limiter.wait_if_needed('gamma_markets')
        url = f"{GAMMA_URL}/markets/slug/{slug}"
        r = session.get(url, timeout=10)
        r.raise_for_status()
        response = r.json()
        
        # ensure we always return a list
        if isinstance(response, list):
            return response
        elif isinstance(response, dict):
            return [response]
        else:
            print(f"Unexpected GAMMA API response type for slug {slug}: {type(response)}")
            return []
            
    except requests.exceptions.Timeout:
        print(f"Timeout fetching GAMMA market for slug: {slug}")
        return []
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Market not found for slug: {slug}")
        else:
            print(f"HTTP error fetching GAMMA market for slug {slug}: {e}")
        return []
    except Exception as e:
        print(f"Error fetching GAMMA market for slug {slug}: {e}")
        return []

def check_markets_closure(session, sharps_cache):
    """Check all tracked markets for closure using GAMMA API."""
    markets_to_check = sharps_cache.get_markets_to_check()
    
    if not markets_to_check:
        return
    
    print(f"Checking {len(markets_to_check)} markets for closure via GAMMA API...")
    
    for market_data in markets_to_check:
        condition_id = market_data['condition_id']
        slug = market_data.get('slug')
        
        if not slug:
            continue
        
        # fetch market data from GAMMA API
        markets = fetch_markets_by_slug(session, slug)
        
        if not markets:
            continue
        
        # find the matching market by condition_id
        matching_market = None
        for market in markets:
            if market.get('conditionId') == condition_id:
                matching_market = market
                break
        
        if not matching_market:
            continue
        
        # check if market is closed
        is_closed = matching_market.get('closed', False)
        
        if is_closed:
            print(f"📊 Market closed detected: {matching_market.get('question', 'Unknown')} ({slug})")
            
            # mark market as closed in database
            sharps_cache.mark_market_closed(condition_id)
            
            # get outcome prices to finalize trades
            outcome_prices_str = matching_market.get('outcomePrices', '[]')
            try:
                outcome_prices = json.loads(outcome_prices_str) if isinstance(outcome_prices_str, str) else outcome_prices_str
            except json.JSONDecodeError as e:
                print(f"Error parsing outcome prices for {slug}: {e}")
                outcome_prices = []
            
            # get all open trades for this market
            conn = sharps_cache._get_conn()
            cur = conn.execute('''
                SELECT id, outcome_idx, buy_price, shares_bought, market_title, outcome_label, current_price
                FROM trades 
                WHERE condition_id = ? AND status = 'OPEN'
            ''', (condition_id,))
            
            open_trades = cur.fetchall()
            
            for trade in open_trades:
                outcome_idx = trade['outcome_idx']
                final_price = None
                
                # get final price for this outcome
                if outcome_idx < len(outcome_prices):
                    try:
                        final_price = float(outcome_prices[outcome_idx])
                        print(f"🎯 Finalizing trade #{trade['id']} ({trade['market_title']} - {trade['outcome_label']}) at {final_price:.3f}")
                        
                        sharps_cache.finalize_trade(
                            trade['id'],
                            final_price,
                            trade['buy_price'],
                            trade['shares_bought']
                        )
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing outcome price for trade #{trade['id']}: {e}")
                        final_price = None
                else:
                    final_price = None

                if final_price is None:
                    fallback_price = trade['current_price'] if trade['current_price'] is not None else trade['buy_price']
                    try:
                        final_price = float(fallback_price)
                        print(f"⚠️  No outcome price for trade #{trade['id']} - using fallback price {final_price:.3f}")
                        sharps_cache.finalize_trade(
                            trade['id'],
                            final_price,
                            trade['buy_price'],
                            trade['shares_bought']
                        )
                    except (ValueError, TypeError) as e:
                        print(f"Error using fallback price for trade #{trade['id']}: {e}")

            # mark any WATCH anchors as WATCHED when market closes
            def _mark_watched(c):
                c.execute('''
                    UPDATE trades
                    SET status = 'WATCHED',
                        current_price = ?
                    WHERE condition_id = ? AND status = 'WATCH'
                ''', (final_price, condition_id))
                return True

            sharps_cache._write_sync(_mark_watched)

# --- API FETCHING LOGIC ---
def fetch_all_positions(session, endpoint, params):
    """Fetch positions with proper error handling."""
    seen_assets = set()
    out = []
    offset = 0
    LIMITS = {'positions': 500, 'closed-positions': 50}
    
    while True:
        p = dict(params)
        p.update({"limit": LIMITS[endpoint], "offset": offset, "sortBy": 'TITLE'})
        try:
            # use specific positions endpoint rate limit
            rate_limit_key = 'data_positions' if endpoint == 'positions' else 'data_general'
            limiter.wait_if_needed(rate_limit_key)
            r = session.get(f"{DATA_URL}/v1/{endpoint}", params=p, timeout=15)
            
            if r.status_code != 200:
                print(f"Warning: {endpoint} returned status {r.status_code}")
                break
            
            batch = r.json()
            if not batch: 
                break

            for item in batch:
                asset = item.get("asset")
                if asset not in seen_assets: 
                    seen_assets.add(asset)
                    out.append(item)

            if len(batch) < LIMITS[endpoint]: 
                break
            
            offset += len(batch)
            
        except requests.exceptions.Timeout:
            print(f"Timeout fetching {endpoint} at offset {offset}")
            break
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error fetching {endpoint}: {e}")
            break
        except Exception as e:
            print(f"Error fetching {endpoint}: {e}")
            break
            
    return out

def get_open_positions_filtered(session, wallet, cached):
    positions = fetch_all_positions(session, "positions", {"user": wallet, 'redeemable': False, "mergeable": False})
    positions.extend(fetch_all_positions(session, "positions", {"user": wallet, 'redeemable': False, "mergeable": True}))
    if not cached:
        redeemable = fetch_all_positions(session, "positions", {"user": wallet, "redeemable": True, "mergeable": False})
        redeemable.extend(fetch_all_positions(session, "positions", {"user": wallet, "redeemable": True, "mergeable": True}))
        positions.extend(redeemable)

    num_open = len(positions)
    filtered, unredeemed = [], []
    for p in positions:
        title = p.get("title", "")
        if not filter_title(title): 
            continue
        if not 0.03 < p['curPrice'] < 0.98:
            unredeemed.append(p)
        else:
            filtered.append(p)
    return filtered, unredeemed, num_open


def get_closed_positions_filtered(session, wallet):
    positions = fetch_all_positions(session, "closed-positions", {"user": wallet})
    filtered = []
    for p in positions:
        if not filter_title(p.get("title", "")): continue
        size = float(p.get("totalBought", 0))
        avg_price = float(p.get("avgPrice", 0))
        if size == 0 or not avg_price < 0.97: continue
        filtered.append(p)
    return filtered

def split_weights_by_shares(positions):
    markets = defaultdict(list)
    
    for pos in positions:
        m_id = pos.get("conditionId")
        if m_id is None:
            continue
        markets[m_id].append(pos)
    
    weighted_positions = []
    
    for m_id, market_positions in markets.items():
        total_shares = sum(float(p.get("size", 0)) for p in market_positions)
        
        if total_shares == 0:
            continue
        
        for pos in market_positions:
            shares = float(pos.get("size", 0))
            if shares > 0:
                weighted_pos = pos.copy()
                weighted_pos["weight_fraction"] = shares / total_shares
                weighted_positions.append(weighted_pos)
    
    return weighted_positions


def calculate_sharpness(positions):
    if not positions:
        return {'sharpness_score': 0, 'mean_adjusted_roi': 0, 't_stat': 0, 'n': 0}

    # extraction & ROI calculation
    avgs = np.array([float(p.get('avgPrice', 0)) for p in positions])
    curs = np.array([float(p.get('curPrice', 0)) for p in positions])
    
    # filter out zeros to avoid division by zero
    mask = avgs > 0
    if not np.any(mask):
        return {'sharpness_score': 0, 'mean_adjusted_roi': 0, 't_stat': 0, 'n': 0}
    
    # calculate net returns (5% gain is 0.05)
    returns = (curs[mask] / avgs[mask]) - 1
    n = len(returns)

    if n < 5: 
        return {'sharpness_score': 0, 'mean_adjusted_roi': round(np.mean(returns), 4), 't_stat': 0, 'n': n}

    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)

    if std_r == 0:
        score = 1.0 if mean_r > 0 else 0
        return {'sharpness_score': score, 'mean_adjusted_roi': round(mean_r, 4), 't_stat': 0, 'n': n}

    # standard T-Statistic
    # "how many standard deviations is our mean away from zero?"
    t_stat = mean_r / (std_r / math.sqrt(n))

    snr = abs(mean_r) / std_r
    k = 10 / max(0.1, snr)  
    shrink = math.sqrt(n / (n + k))

    sharpness = max(0, (t_stat * shrink) / 2) # scaling by 2 for a readable range

    return {
        'sharpness_score': round(sharpness, 3), 
        'mean_adjusted_roi': round(mean_r, 4), 
        't_stat': round(t_stat, 3), 
        'n': n
    }


def analyze_user_positions(session, user, sharps_cache):
    wallet = user['proxyWallet']
    username = user.get('userName', 'Unknown')
    cached = sharps_cache.get(wallet)

    if cached is not None and float(cached['sharpness_score']) < 0.1 and int(cached['num_open']) < 1000:
        return

    open_pos_raw, unredeemed, num_open = get_open_positions_filtered(session, wallet, cached is not None)
    open_pos_weighted = split_weights_by_shares(open_pos_raw)
    open_positions = [{
        "conditionId": p['conditionId'], 
        "size": p['size'], 
        "avgPrice": p['avgPrice'],
        "totalBought": p['totalBought'], 
        "initialValue": p['initialValue'],
        "realizedPnl": p['realizedPnl'],
        "title": p['title'], 
        "outcome": p['outcome'], 
        "outcomeIndex": p['outcomeIndex'],
        "asset": p.get('asset'),
        "slug": p.get('slug'),  # store slug from position
        "weight_fraction": p.get('weight_fraction', 1.0)  # store the weight fraction
    } for p in open_pos_weighted]

    if cached:
        if cached.get('username') is None: cached['username'] = username
        cached.update({'open_positions': open_positions, 'num_open': num_open})
        sharps_cache.update_sharp(wallet, cached)
        return cached

    closed_positions = get_closed_positions_filtered(session, wallet)
    closed_positions.extend(unredeemed)

    metrics = calculate_sharpness(closed_positions)
    
    result = {
        'username': username, 'wallet': wallet, 
        'sharpness_score': round(metrics['sharpness_score'], 4),
        'num_open': num_open, 'open_positions': open_positions
    }
    sharps_cache.update_sharp(wallet, result)
    print(f"{username[:40]:<40} Sharp={metrics['sharpness_score']:.4f}")    
    return result


def calculate_edge(w_for, w_against):
    total_w = w_for + w_against
    if total_w == 0:
        return 0.0, 0.5, 0.0
    dominance = (w_for - w_against) / total_w
    edge_score = (w_for - w_against) * abs(dominance)
    likelihood = (w_for + 1) / (total_w + 2)
    return round(edge_score, 4), round(likelihood, 4), round(dominance, 4)


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class RNNModel(torch.nn.Module):
    # defined the same as in training
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn_query = torch.nn.Parameter(torch.randn(hidden_dim))
        self.attn_fc = torch.nn.Linear(hidden_dim, hidden_dim)
        self.pool_proj = torch.nn.Linear(hidden_dim * 3, hidden_dim)
        self.head_ln = torch.nn.LayerNorm(hidden_dim)
        self.head_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout)
        )
        self.head_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        lengths_dev = lengths.to(out.device)
        max_len = out.size(1)
        mask = (torch.arange(max_len, device=out.device).unsqueeze(0) < lengths_dev.unsqueeze(1)).unsqueeze(-1)

        scores = torch.tanh(self.attn_fc(out))
        scores = torch.matmul(scores, self.attn_query)
        scores = scores.masked_fill(~mask.squeeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)

        avg_pool = (out * mask).sum(dim=1) / lengths_dev.clamp_min(1).unsqueeze(1).to(out.dtype)
        max_val = (out.masked_fill(~mask, -1e9)).max(dim=1).values

        combined = torch.cat([avg_pool, max_val, context], dim=1)
        h = self.pool_proj(combined)
        h = self.head_ln(h)
        h = h + self.head_mlp(h)
        logits = self.head_out(h).squeeze(-1)
        return logits


def _esport_flag(title):
    """Returns True if esport"""
    title = (title or "").lower()
    return 1 if ("lol" in title or "dota 2" in title or "counter-strike" in title) else 0


def build_step_features(row, first_ts, first_price, market_title):
    if not isinstance(row, list) or len(row) < 6:
        return None
    try:
        bid_price, _ask_price, edge, w_for, w_against, ts = row[:6]
        price = float(bid_price)
        edge = float(edge)
        w_for = float(w_for)
        w_against = float(w_against)
        ts = float(ts)
    except Exception:
        return None

    n_sharps_for = row[6] if len(row) > 6 else None
    n_sharps_against = row[7] if len(row) > 7 else None
    avg_sharp_for = row[8] if len(row) > 8 else None
    avg_sharp_against = row[9] if len(row) > 9 else None
    avg_price_for = row[10] if len(row) > 10 else None
    avg_price_against = row[11] if len(row) > 11 else None
    esport = row[13] if len(row) > 13 else None

    if esport is None:
        esport = _esport_flag(market_title)

    def logit(p):
        p = min(max(float(p), 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    delta_price_for = (price - float(avg_price_for)) if avg_price_for is not None else 0.0
    delta_price_against = (price - float(avg_price_against)) if avg_price_against is not None else 0.0

    feats = [
        price,
        logit(price),
        logit(first_price),
        logit(avg_price_for) if avg_price_for is not None else 0.0,
        logit(avg_price_against) if avg_price_against is not None else 0.0,
        logit(min(max(delta_price_for + 0.5, 0.0), 1.0)),
        logit(min(max(delta_price_against + 0.5, 0.0), 1.0)),
        logit(min(max(price - first_price + 0.5, 0.0), 1.0)),
        edge,
        edge / max(price, 0.01),
        w_for,
        w_for - w_against,
        ts - first_ts,                     # dt
        first_price,                       # first_price
        price - first_price,               # price_change_from_first
        float(n_sharps_for) if n_sharps_for is not None else 0.0,
        float(n_sharps_against) if n_sharps_against is not None else 0.0,
        float(avg_sharp_for) if avg_sharp_for is not None else 0.0,
        float(avg_sharp_against) if avg_sharp_against is not None else 0.0,
        float(avg_price_for) if avg_price_for is not None else 0.0,
        float(avg_price_against) if avg_price_against is not None else 0.0,
        delta_price_for,
        delta_price_against,
        # float(esport),
    ]
    return feats


def build_sequence_from_history(history, market_title, min_len=100, cutoff_low=0.01, cutoff_high=0.99, price_min=0.05, price_max=0.95):
    if not history or len(history) < 50:
        return None, None

    first_ts = None
    first_price = None
    seq = []
    price_seq = []
    for row in history:
        if not isinstance(row, list) or len(row) < 6:
            continue
        price_val = float(row[0])
        if first_ts is None:
            first_ts = row[5]
            first_price = row[0]
        if price_val <= cutoff_low or price_val >= cutoff_high:
            break
        feats = build_step_features(row, first_ts, first_price, market_title)
        if feats is None:
            continue
        seq.append(feats)
        price_seq.append(price_val)

    if len(seq) < min_len:
        return None, None
    seq = seq[-min_len:]
    price_seq = price_seq[-min_len:]
    last_price = price_seq[-1]
    if last_price < price_min or last_price > price_max:
        return None, None
    return np.array(seq, dtype=np.float32), last_price


def predict_from_history(history, model, mean, std, market_title):
    seq, last_price = build_sequence_from_history(history, market_title)
    if seq is None or last_price is None:
        return None
    if mean is None or std is None or mean.size != seq.shape[1] or std.size != seq.shape[1]:
        return None

    seq = (seq - mean) / std
    device = next(model.parameters()).device
    x = torch.from_numpy(seq).unsqueeze(0).to(device)
    lengths = torch.tensor([seq.shape[0]], dtype=torch.long, device=device)

    with torch.inference_mode():
        logits = model(x, lengths)
        p_model = torch.sigmoid(logits).item()
    return float(p_model)


def predict_batch_from_histories(items, model, mean, std):
    """
    items: list of (key, history, market_title)
    returns: dict key -> probability
    """
    if model is None or mean is None or std is None:
        return {}

    sequences = []
    keys = []
    for key, history, title in items:
        seq, _last_price = build_sequence_from_history(history, title)
        if seq is None or _last_price is None:
            continue
        if mean.size != seq.shape[1] or std.size != seq.shape[1]:
            continue
        seq = (seq - mean) / std
        sequences.append(seq)
        keys.append(key)

    if not sequences:
        return {}

    device = next(model.parameters()).device
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long, device=device)
    max_len = max(lengths).item()
    feat_dim = sequences[0].shape[1]
    x = torch.zeros(len(sequences), max_len, feat_dim, dtype=torch.float32, device=device)
    for i, seq in enumerate(sequences):
        x[i, :seq.shape[0], :] = torch.from_numpy(seq).to(device)

    with torch.inference_mode():
        logits = model(x, lengths)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()

    out = {}
    for key, p in zip(keys, probs):
        out[key] = float(p)
    return out


def print_pnl_summary(sharps_cache):
    """Print combined total PNL and ROI statistics."""
    try:
        conn = sharps_cache._get_conn()
        
        # get all trades
        cur = conn.execute("SELECT status, pnl, buy_price, pnl_per_share, shares_bought, buy_edge FROM trades")
        trades = cur.fetchall()
        
        if not trades:
            return
        
        realized_pnl = 0
        unrealized_pnl = 0
        invested_closed = 0
        invested_current = 0
        num_open = 0
        num_closed = 0
        total_shares = 0
        total_edge = 0
        
        for trade in trades:
            if trade['pnl_per_share'] is not None and abs(trade['pnl_per_share']) > 0.03:
                investment = trade['buy_price'] * trade['shares_bought']
                pnl = trade['pnl'] or 0
                
                # count all shares for avg_pnl_per_share calculation
                total_shares += trade['shares_bought']
                
                if trade['status'] == 'CLOSED':
                    realized_pnl += pnl
                    invested_closed += investment
                    num_closed += 1
                elif trade['status'] == 'OPEN':
                    unrealized_pnl += pnl
                    invested_current += investment
                    num_open += 1
                    total_edge += trade['buy_edge'] or 0
        
        total_pnl = realized_pnl + unrealized_pnl
        total_invested = invested_closed + invested_current
        roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        realized_roi = (realized_pnl / invested_closed * 100) if invested_closed > 0 else 0
        avg_edge = (total_edge / num_open) if num_open > 0 else 0
        avg_pnl_per_share = (total_pnl / total_shares) if total_shares > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"💰 INVESTMENT: ${total_invested:.2f} | TOTAL PNL: ${total_pnl:.2f} | ROI: {roi:.2f}%")
        print(f"📊 REALIZED: ${realized_pnl:.2f} ({realized_roi:.2f}%) | UNREALIZED: ${unrealized_pnl:.2f}")
        if LAST_LOOP_TIMINGS:
            print(f"⏱️  Timings: lb={LAST_LOOP_TIMINGS.get('leaderboard', 0):.2f}s | refresh={LAST_LOOP_TIMINGS.get('refresh_positions', 0):.2f}s | closure={LAST_LOOP_TIMINGS.get('closure_check', 0):.2f}s | process={LAST_LOOP_TIMINGS.get('process_edges', 0):.2f}s")
        if LAST_PROCESS_TIMINGS:
            print(f"⏱️  Process: infer={LAST_PROCESS_TIMINGS.get('inference', 0):.2f}s | non-infer={LAST_PROCESS_TIMINGS.get('non_inference', 0):.2f}s")
        print(f"{'='*60}")
        
        # record PNL snapshot for historical tracking
        sharps_cache.record_pnl_snapshot({
            'total_investment': total_invested,
            'total_pnl': total_pnl,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'avg_pnl_per_share': avg_pnl_per_share,
            'roi': roi,
            'num_open': num_open,
            'num_closed': num_closed,
            'avg_edge': avg_edge,
            'total_shares': total_shares
        })
        
    except Exception as e:
        print(f"Error calculating PNL summary: {e}")

def fetch_orderbook_prices(session, token_requests):
    """
    Fetch orderbook prices for multiple tokens with proper error handling.
    Returns tuple: (prices_dict, success_bool)
    """
    if not token_requests:
        return {}, True
    
    # validate token IDs before sending
    valid_requests = []
    for req in token_requests:
        token_id = req.get('token_id')
        if token_id and isinstance(token_id, str) and len(token_id) > 0:
            valid_requests.append(req)
    
    if not valid_requests:
        return {}, True
    
    batch_size = 500
    all_prices = {}
    
    for i in range(0, len(valid_requests), batch_size):
        batch = valid_requests[i:i + batch_size]
        
        try:
            limiter.wait_if_needed('clob_market')
            r = session.post(
                f"{CLOB_URL}/prices", 
                json=batch, 
                headers={'Content-Type': 'application/json'}, 
                timeout=15
            )
            
            # only use data if we got a successful response
            if r.status_code == 200:
                batch_prices = r.json()
                if isinstance(batch_prices, dict):
                    all_prices.update(batch_prices)
            else:
                print(f"Warning: orderbook prices returned status {r.status_code} for batch {i//batch_size + 1}")
                return all_prices, False  # signal partial failure
                
        except requests.exceptions.Timeout:
            print(f"Timeout fetching orderbook prices for batch {i//batch_size + 1}")
            return all_prices, False
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error fetching orderbook prices: {e}")
            return all_prices, False
        except Exception as e:
            print(f"Exception fetching orderbook prices: {e}")
            return all_prices, False
    
    return all_prices, True

def process_live_edges_and_trades(session, sharps_cache, model, model_mean, model_std):
    infer_time = 0.0
    cached_users = sharps_cache.get_all_cached_users()
    market_forecasts = defaultdict(list)
    market_sharpness = defaultdict(lambda: defaultdict(list))
    market_avg_prices = defaultdict(lambda: defaultdict(list))
    market_names = {}
    market_labels = defaultdict(dict)
    market_token_ids = defaultdict(dict)
    market_slugs = {}  # store slugs for each market

    # 1. AGGREGATE SHARP POSITIONS
    for row in cached_users:
        data = dict(row)
        sharpness = float(data['sharpness_score'])
        if sharpness <= 0.5: continue
        
        positions = json.loads(data['open_positions'])

        for pos in positions:
            m_id = pos.get("conditionId")
            idx = int(pos.get("outcomeIndex"))
            weight_fraction = float(pos.get("weight_fraction", 1.0))
            
            # apply the weight fraction to split the sharp's weight across outcomes
            market_forecasts[m_id].append({
                "outcome_index": idx,
                "entry_price": float(pos.get("avgPrice", 0)),
                "weight": sharpness * weight_fraction  # split weight proportionally
            })
            market_sharpness[m_id][idx].append(sharpness)
            try:
                market_avg_prices[m_id][idx].append(float(pos.get("avgPrice", 0)))
            except Exception:
                pass
            market_names[m_id] = pos.get("title") or market_names.get(m_id, "Unknown")
            market_labels[m_id][idx] = pos.get("outcome") or market_labels[m_id].get(idx, str(idx))
            
            # store slug if available
            if pos.get("slug"):
                market_slugs[m_id] = pos.get("slug")
            
            if pos.get("asset"):
                market_token_ids[m_id][idx] = str(pos.get("asset"))
            if pos.get("oppositeAsset"):
                market_token_ids[m_id][1 - idx] = str(pos.get("oppositeAsset"))

    active_trades = sharps_cache.get_active_trades()
    monitored_trades = sharps_cache.get_all_monitored_trades()
    
    # active_trades is now a dict: {condition_id: [trade1, trade2]}
    for m_id, trades in active_trades.items():
        for trade in trades:
            idx = trade['outcome_idx']
            
            # add token_id if not already present
            if trade.get('token_id'):
                if m_id not in market_token_ids:
                    market_token_ids[m_id] = {}
                if idx not in market_token_ids[m_id]:
                    market_token_ids[m_id][idx] = trade['token_id']
            
            # preserve outcome labels
            if trade.get('outcome_label'):
                if m_id not in market_labels:
                    market_labels[m_id] = {}
                if idx not in market_labels[m_id]:
                    market_labels[m_id][idx] = trade['outcome_label']
            
            # ensure market name is preserved
            if trade.get('market_title') and m_id not in market_names:
                market_names[m_id] = trade['market_title']

    # 2. SYNC ACTIVE TRADES & PRICES
    active_trade_map = active_trades  # now already grouped by condition_id
    
    # collect all tokens from forecasts and monitored trades
    all_token_ids = set()
    for m_id, outcomes in market_token_ids.items():
        for token_id in outcomes.values(): 
            if token_id:  # Validate token ID exists
                all_token_ids.add(token_id)
    
    for trade in monitored_trades:
        if trade.get('token_id'): 
            all_token_ids.add(trade['token_id'])
    
    # build token requests with validation
    token_requests = []
    for t_id in all_token_ids:
        if t_id and isinstance(t_id, str):
            token_requests.extend([
                {"token_id": t_id, "side": "SELL"}, 
                {"token_id": t_id, "side": "BUY"}
            ])
    
    orderbook_prices, fetch_success = fetch_orderbook_prices(session, token_requests)
    
    # if fetch failed, skip trading logic to avoid bad data
    if not fetch_success:
        print("Orderbook fetch failed - skipping trade updates this cycle")
        return
    
    market_asks = defaultdict(dict)  # Price to BUY
    market_bids = defaultdict(dict)  # Price to SELL
    
    # map orderbook response back to markets
    for m_id, outcomes in market_token_ids.items():
        for idx, t_id in outcomes.items():
            if t_id in orderbook_prices:
                try:
                    if ask := orderbook_prices[t_id].get("SELL"): 
                        market_asks[m_id][idx] = float(ask)
                    if bid := orderbook_prices[t_id].get("BUY"): 
                        market_bids[m_id][idx] = float(bid)
                except (ValueError, TypeError): 
                    print('problem')
                    pass

    # track market spreads for filtering
    for m_id, outcomes in market_token_ids.items():
        asks = market_asks.get(m_id, {})
        bids = market_bids.get(m_id, {})
        
        # calculate spreads for both tokens (outcomes)
        token_spreads = {}
        for idx in outcomes.keys():
            ask = asks.get(idx)
            bid = bids.get(idx)
            if ask is not None and bid is not None and ask > 0 and bid > 0:
                token_spreads[idx] = ask - bid
        
        # update spread tracking if we have both tokens
        if len(token_spreads) >= 2:
            sorted_indices = sorted(token_spreads.keys())
            if len(sorted_indices) >= 2:
                sharps_cache.update_market_spread(
                    m_id,
                    outcomes.get(sorted_indices[0]),
                    outcomes.get(sorted_indices[1]),
                    token_spreads.get(sorted_indices[0], 0),
                    token_spreads.get(sorted_indices[1], 0),
                    market_names.get(m_id),
                    market_slugs.get(m_id)  # pass slug to be stored
                )

    all_monitored_ids = set(market_forecasts.keys()) | set(active_trade_map.keys())

    # batch inference for all watch anchors (one per market)
    p_model_by_market = {}
    if model is not None:
        batch_items = []
        for m_id in all_monitored_ids:
            trades = active_trade_map.get(m_id, [])
            watch_trades = [t for t in trades if t.get('status') == 'WATCH']
            if not watch_trades:
                continue
            anchor = watch_trades[0]
            try:
                hist = json.loads(anchor.get("history") or "[]")
            except Exception:
                hist = []
            batch_items.append((m_id, hist, market_names.get(m_id)))
        if batch_items:
            t_inf_start = time.time()
            p_model_by_market = predict_batch_from_histories(batch_items, model, model_mean, model_std)
            infer_time += time.time() - t_inf_start

    for m_id in all_monitored_ids:
        forecasts = market_forecasts.get(m_id, [])
        trades = active_trade_map.get(m_id, [])  
        
        current_asks = market_asks.get(m_id, {})
        current_bids = market_bids.get(m_id, {})
        
        # check if market is marked as closed - skip trading logic if so
        market_stats = sharps_cache.get_market_spread_stats(m_id)
        if market_stats and market_stats.get('is_closed'):
            continue
        
        # determine weights for all outcomes (no penalties)
        weights = defaultdict(float)
        for f in forecasts:
            idx = f["outcome_index"]
            weights[idx] += float(f["weight"])

        # calculate Best Outcome based on sharp sentiment (still used for WATCH creation)
        best_idx = max(weights, key=weights.get) if weights else None
        w_for_best = weights.get(best_idx, 0)
        w_against_best = sum(w for i, w in weights.items() if i != best_idx)
        best_edge, _, _ = calculate_edge(w_for_best, w_against_best)

        # --- MANAGE EXISTING TRADES / WATCH ANCHOR ---
        open_trades = [t for t in trades if t.get('status') == 'OPEN']
        watch_trades = [t for t in trades if t.get('status') == 'WATCH']
        anchor_trade = watch_trades[0] if watch_trades else None

        # get shares for both sides to calculate targets (OPEN trades only)
        shares_by_idx = {}
        for trade in open_trades:
            shares_by_idx[trade['outcome_idx']] = trade['shares_bought']

        if anchor_trade and model is not None:
            held_idx = anchor_trade['outcome_idx']
            opposite_idx = 1 - held_idx
            p_model = p_model_by_market.get(m_id)

            ask_held = current_asks.get(held_idx)
            bid_held = current_bids.get(held_idx)
            ask_opp = current_asks.get(opposite_idx)
            bid_opp = current_bids.get(opposite_idx)

            if p_model is not None and ask_held and bid_held and ask_opp and bid_opp:
                # expected value vs price 
                model_edge_held = p_model - ask_held
                model_edge_opp = (1 - p_model) - ask_opp

                existing_trade_held = next((t for t in open_trades if t["outcome_idx"] == held_idx), None)
                existing_trade_opp = next((t for t in open_trades if t["outcome_idx"] == opposite_idx), None)

                current_this_side = shares_by_idx.get(held_idx, 0)
                current_opposite = shares_by_idx.get(opposite_idx, 0)

                target_this_side = max(0, int(round(model_edge_held * MODEL_EDGE_SCALE)))
                target_opposite = max(0, int(round(model_edge_opp * MODEL_EDGE_SCALE)))

                # step 2: enter/scale this side to target based on EV
                shares_to_add = max(0, target_this_side - current_this_side)
                if model_edge_held > MIN_EV_THRESHOLD and shares_to_add > 0:
                    current_spread = (ask_held - bid_held) if (ask_held and bid_held) else 1.0
                    spread_ok = sharps_cache.should_trade_market(m_id, current_spread)
                    if ask_held and bid_held and spread_ok and MIN_BUY_PRICE < ask_held < MAX_BUY_PRICE and (ask_held - bid_held) <= 0.03:
                        if existing_trade_held:
                            total_shares = current_this_side + shares_to_add
                            new_avg = (
                                (current_this_side * existing_trade_held['buy_price']) +
                                (shares_to_add * ask_held)
                            ) / total_shares
                            # sharps_cache.scale_trade(
                            #     existing_trade_held['id'],
                            #     total_shares,
                            #     round(new_avg, 4),
                            #     model_edge_held
                            # )
                        else:
                            buy_w_for = weights.get(held_idx, 0)
                            buy_w_against = sum(w for i, w in weights.items() if i != held_idx)
                            sharps_cache.open_trade({
                                'condition_id': m_id,
                                'outcome_idx': held_idx,
                                'token_id': market_token_ids[m_id].get(held_idx),
                                'outcome_label': market_labels[m_id].get(held_idx),
                                'buy_price': ask_held,
                                'buy_edge': model_edge_held,
                                'shares_bought': shares_to_add,
                                'market_title': market_names.get(m_id),
                                'sharps_for': buy_w_for,
                                'sharps_against': buy_w_against,
                                'np_sharps_for': buy_w_for,
                                'np_sharps_against': buy_w_against,
                                'current_ev': model_edge_held
                            })
                        shares_by_idx[held_idx] = current_this_side + shares_to_add

                # step 3: enter/scale opposite side to target based on EV
                shares_to_add = max(0, target_opposite - current_opposite)
                if model_edge_opp > MIN_EV_THRESHOLD and shares_to_add > 0:
                    current_spread = (ask_opp - bid_opp) if (ask_opp and bid_opp) else 1.0
                    spread_ok = sharps_cache.should_trade_market(m_id, current_spread)
                    if ask_opp and bid_opp and spread_ok and MIN_BUY_PRICE < ask_opp < MAX_BUY_PRICE and (ask_opp - bid_opp) <= 0.03:
                        if existing_trade_opp:
                            total_shares = current_opposite + shares_to_add
                            new_avg = (
                                (current_opposite * existing_trade_opp['buy_price']) +
                                (shares_to_add * ask_opp)
                            ) / total_shares
                            # sharps_cache.scale_trade(
                            #     existing_trade_opp['id'],
                            #     total_shares,
                            #     round(new_avg, 4),
                            #     model_edge_opp
                            # )
                        else:
                            buy_w_for = weights.get(opposite_idx, 0)
                            buy_w_against = sum(w for i, w in weights.items() if i != opposite_idx)
                            sharps_cache.open_trade({
                                'condition_id': m_id,
                                'outcome_idx': opposite_idx,
                                'token_id': market_token_ids[m_id].get(opposite_idx),
                                'outcome_label': market_labels[m_id].get(opposite_idx),
                                'buy_price': ask_opp,
                                'buy_edge': model_edge_opp,
                                'shares_bought': shares_to_add,
                                'market_title': market_names.get(m_id),
                                'sharps_for': buy_w_for,
                                'sharps_against': buy_w_against,
                                'np_sharps_for': buy_w_for,
                                'np_sharps_against': buy_w_against,
                                'current_ev': model_edge_opp
                            })
                        shares_by_idx[opposite_idx] = current_opposite + shares_to_add

        # --- ENTER NEW WATCH ANCHOR ---
        # only watch if we have no trades on this condition yet
        total_weight = w_for_best + w_against_best
        if not trades and best_idx is not None and total_weight > WATCH_THRESHOLD:
            entry_p = current_asks.get(best_idx)
            exit_p = current_bids.get(best_idx)
            
            # calculate current spread
            current_spread = (entry_p - exit_p) if (entry_p and exit_p) else 1.0
            
            # check spread criteria before entering
            spread_ok = sharps_cache.should_trade_market(m_id, current_spread)
            
            # ensure liquidity (tight spread) and safe price range (unresolved market)
            if entry_p and exit_p and spread_ok and (entry_p - exit_p) <= 0.03 and 0.05 < entry_p < 0.95:
                sharps_cache.open_watch_trade({
                    'condition_id': m_id, 
                    'outcome_idx': best_idx, 
                    'token_id': market_token_ids[m_id].get(best_idx),
                    'outcome_label': market_labels[m_id].get(best_idx), 
                    'buy_price': entry_p, 
                    'buy_edge': best_edge,  # store edge frozen at time of watching
                    'shares_bought': 0,
                    'market_title': market_names.get(m_id),
                    'sharps_for': w_for_best,
                    'sharps_against': w_against_best,
                    'np_sharps_for': w_for_best,
                    'np_sharps_against': w_against_best
                })
        
        # update prices and current edge for all OPEN and WATCH positions
        open_outcomes = {t['outcome_idx'] for t in open_trades}
        watch_outcomes = {t['outcome_idx'] for t in watch_trades}
        outcomes_to_update = open_outcomes | watch_outcomes
        for outcome_idx in outcomes_to_update:
            bid_price = current_bids.get(outcome_idx)
            ask_price = current_asks.get(outcome_idx)
            if bid_price:
                # calculate current edge for this outcome
                w_for = weights.get(outcome_idx, 0)
                w_against = sum(w for i, w in weights.items() if i != outcome_idx)
                current_edge, _, _ = calculate_edge(w_for, w_against)

                sharps_for_list = market_sharpness.get(m_id, {}).get(outcome_idx, [])
                sharps_against_list = []
                for idx, lst in market_sharpness.get(m_id, {}).items():
                    if idx != outcome_idx:
                        sharps_against_list.extend(lst)

                n_sharps_for = len(sharps_for_list)
                n_sharps_against = len(sharps_against_list)
                avg_sharp_for = float(np.mean(sharps_for_list)) if sharps_for_list else 0.0
                avg_sharp_against = float(np.mean(sharps_against_list)) if sharps_against_list else 0.0
                prices_for = market_avg_prices.get(m_id, {}).get(outcome_idx, [])
                avg_price_for = float(np.mean(prices_for)) if prices_for else None
                prices_against = [
                    p for idx, plist in market_avg_prices.get(m_id, {}).items()
                    if idx != outcome_idx for p in plist
                ]
                avg_price_against = float(np.mean(prices_against)) if prices_against else None

                # fallback to current ask if no sharp-held avgPrice exists
                if avg_price_for is None:
                    avg_price_for = current_asks.get(outcome_idx)
                if avg_price_against is None:
                    other_asks = [
                        current_asks.get(i) for i in current_asks.keys()
                        if i != outcome_idx and current_asks.get(i) is not None
                    ]
                    if other_asks:
                        avg_price_against = float(np.mean(other_asks))

                # model EV if available
                current_ev = None
                title = (market_names.get(m_id) or "")
                title_lower = title.lower()
                esport = 1 if ("lol" in title_lower or "dota 2" in title_lower or "counter-strike" in title_lower) else 0
                if model is not None and watch_trades:
                    anchor = watch_trades[0]
                    p_model = p_model_by_market.get(m_id)
                    if p_model is not None:
                        # bid-based EV (match training price)
                        if outcome_idx == anchor.get("outcome_idx"):
                            current_ev = p_model - bid_price
                        else:
                            current_ev = (1 - p_model) - bid_price

                sharps_cache.update_trade_prices(
                    m_id,
                    outcome_idx,
                    bid_price,
                    ask_price,
                    current_edge,
                    w_for,
                    w_against,
                    n_sharps_for,
                    n_sharps_against,
                    avg_sharp_for,
                    avg_sharp_against,
                    avg_price_for,
                    avg_price_against,
                    current_ev,
                    esport,
                )

    LAST_PROCESS_TIMINGS.clear()
    LAST_PROCESS_TIMINGS.update({
        "inference": infer_time
    })

def run_loop():
    """Main trading loop with proper connection management."""
    sharps_cache = TradesDB(DB_FILE, reset_trades=False, reset_sharps=False)
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        input_dim = int(checkpoint.get("input_dim", 0))
        if input_dim <= 0:
            raise ValueError("Invalid model input_dim in checkpoint.")
        state = checkpoint.get("model_state", {})
        if "hidden_dim" in checkpoint:
            hidden_dim = int(checkpoint.get("hidden_dim", 16))
        else:
            w_hh = state.get("rnn.weight_hh_l0")
            hidden_dim = int(w_hh.shape[1]) if w_hh is not None else 16
        num_layers = int(checkpoint.get("num_layers", 1))
        dropout = float(checkpoint.get("dropout", 0.1))
        model = RNNModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model_mean = np.array(checkpoint.get("mean", []), dtype=np.float32)
        model_std = np.array(checkpoint.get("std", []), dtype=np.float32)
        if model_mean.size == 0 or model_std.size == 0:
            raise ValueError("Missing mean/std in checkpoint.")
        print(f"Loaded model: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model {MODEL_PATH}: {e}")
        model = None
        model_mean = None
        model_std = None
    
    session = requests.Session()
    retry_strategy = Retry(
        total=2, 
        backoff_factor=0.5, 
        status_forcelist=[429, 500, 502, 503, 504]
    )
    
    adapter = HTTPAdapter(
        pool_connections=MAX_WORKERS,  
        pool_maxsize=MAX_WORKERS,  
        max_retries=retry_strategy,
        pool_block=False      
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    print(f"Starting Trading Engine.")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        closure_future = None
        while True:
            loop_start = time.time()
            print(f"\n--- Starting Loop at {datetime.now().strftime('%H:%M:%S')} ---")
            
            try:
                t_lb_start = time.time()
                cached_users = sharps_cache.get_all_cached_users()
                leaderboard = []
                
                # fetch leaderboard if cache is smaller than leaderboard
                if len(cached_users) < 500: 
                    print("Fetching leaderboard...")
                    for offset in range(0, LEADERBOARD_SIZE, 50):
                        try:
                            limiter.wait_if_needed('data_general')
                            r = session.get(
                                f"{DATA_URL}/v1/leaderboard", 
                                params={
                                    "category": CATEGORY, 
                                    "limit": 50, 
                                    "offset": offset, 
                                    "timePeriod": "month"
                                }, 
                                timeout=15
                            )
                            if r.status_code == 200: 
                                leaderboard.extend(r.json())
                            else:
                                print(f"Warning: leaderboard returned status {r.status_code}")
                        except requests.exceptions.Timeout:
                            print(f"Timeout fetching leaderboard at offset {offset}")
                        except Exception as e:
                            print(f"Error fetching leaderboard: {e}")
                else:
                    leaderboard = sorted(cached_users, key=lambda x: x.get('num_open', 0), reverse=True)
                t_lb_end = time.time()

                print("Refreshing sharp positions...")
                t_refresh_start = time.time()
                futures = {executor.submit(analyze_user_positions, session, u, sharps_cache): u for u in leaderboard}
                for future in as_completed(futures):
                    try: 
                        future.result()
                    except Exception as e:
                        print(f"Error analyzing user: {e}")
                t_refresh_end = time.time()

                # check for market closures using GAMMA API (non-blocking)
                t_closure_start = time.time()
                if closure_future is None or closure_future.done():
                    if closure_future is not None:
                        try:
                            closure_future.result()
                        except Exception as e:
                            print(f"Error in market closure thread: {e}")
                    closure_future = executor.submit(check_markets_closure, session, sharps_cache)
                t_closure_end = time.time()

                t_process_start = time.time()
                process_live_edges_and_trades(session, sharps_cache, model, model_mean, model_std)
                t_process_end = time.time()

                LAST_LOOP_TIMINGS.clear()
                LAST_LOOP_TIMINGS.update({
                    "leaderboard": t_lb_end - t_lb_start,
                    "refresh_positions": t_refresh_end - t_refresh_start,
                    "closure_check": t_closure_end - t_closure_start,
                    "process_edges": t_process_end - t_process_start,
                })
                LAST_PROCESS_TIMINGS["non_inference"] = max(
                    0.0,
                    (t_process_end - t_process_start) - LAST_PROCESS_TIMINGS.get("inference", 0.0)
                )

                print_pnl_summary(sharps_cache)

                duration = time.time() - loop_start
                print(f"Loop finished in {duration:.2f}s.")
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(5)

if __name__ == "__main__":
    try:
        run_loop()
    except KeyboardInterrupt:
        print("\nStopping script...")

        # history col: (price, edge, w_for, w_against, ts, n_sharps_for, n_sharps_against, avg_sharp_for, avg_sharp_against, avg_price_for, avg_price_against, ev)
