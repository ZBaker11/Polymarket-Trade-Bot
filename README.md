# Polymarket Sports Trading Bot (Prototype)
A Python prototype for monitoring and paper-trading sports prediction markets on Polymarket, built around a "smart money" following strategy backed by a small GRU-based sequence model. The project never made it past the prototype stage; I was never able to manage to prove its profitability. 

# Background & Motivation
Polymarket is a decentralized prediction market platform where users buy outcome shares priced between $0 and $1. A share pays out $1 if its outcome occurs, so prices reflect implied probabilities. The markets are relatively efficient, but not perfectly so. Particularly in sports, sharp bettors (users with a demonstrated edge) can move prices before the broader market catches up.
The core idea here is straightforward: if a cluster of historically profitable traders are holding the same side of a market, that's a signal worth paying attention to. The bot scrapes Polymarket's leaderboard for top monthly performers, scores each user's historical profitability as a "sharpness" score using a shrinkage-adjusted t-statistic, and then tracks their open positions in real time. When enough sharp weight accumulates on one side of a market, it starts watching that market. Once a watch position has built up enough price history, a GRU model runs inference on that sequence and, if expected value clears a minimum threshold, logs a paper trade.
The project was also a way to get comfortable with multi-threaded API polling, SQLite WAL mode under concurrent reads/writes, and deploying a small PyTorch model in an online inference loop.


# Tables

# sharps

wallet — the user's proxy wallet address, used as the primary key

username — display name pulled from the leaderboard response

sharpness_score — the shrinkage-adjusted t-statistic computed from their closed position returns; anything below 0.1 is effectively ignored, and the trading logic only acts on users above 0.5

num_open — total number of open positions across all markets, not just the filtered ones; used to decide whether to re-fetch a cached user's positions

open_positions — JSON array of their current filtered positions, each containing condition ID, outcome index, average entry price, share size, market title, slug, and a weight fraction (their share of total volume on that outcome)

last_checked — unix timestamp of the last time this user's positions were refreshed


# trades

id — autoincrement primary key

condition_id — Polymarket's condition ID for the market; paired with outcome_idx to form a unique constraint on open trades, preventing duplicates

token_id — the ERC-1155 token ID for this specific outcome share, used to query CLOB orderbook prices

market_title — human-readable market name, stored here so it's available without a join

outcome_idx — integer index of the outcome (0 or 1 for binary markets)

outcome_label — the text label for that outcome (e.g. "Yes", "Team A")

status — one of WATCH, OPEN, or CLOSED; the partial unique index on (condition_id, outcome_idx) where status = 'OPEN' prevents entering the same side twice

buy_time — unix timestamp of when the trade or watch was opened

buy_price — ask price at entry, frozen at open time; for WATCH anchors this is the price when the watch started

buy_edge — edge score at entry, also frozen; for OPEN trades this is the model EV at the moment of entry

current_edge — updated each loop as sharp weights shift; diverges from buy_edge as the market moves

buy_weights_for / buy_weights_against — total sharp weight on each side at entry time, stored for reference

shares_bought — number of shares in the position; 0 for WATCH anchors

current_price — most recent bid price, updated every loop

pnl_per_share — current_price - buy_price, updated live

pnl — pnl_per_share * shares_bought

history — JSON array of timestep snapshots, each containing bid price, ask price, edge, sharp weights, sharpness averages, average entry prices on each side, model EV, and an esport flag; this is what gets fed to the GRU

current_ev — the most recent model EV estimate, updated each time inference runs


# markets

condition_id — primary key, Polymarket's condition ID

market_title — stored here in case it's not available from the sharps data

slug — the URL slug used to query the GAMMA API for closure status

token0 / token1 — token IDs for each outcome, mirroring what's in trades but at the market level

token0_spread / token1_spread — exponential moving average (alpha = 0.2) of the bid-ask spread for each outcome token, updated every loop; used by should_trade_market() to filter out illiquid markets

num_checks — how many times this market's spread has been recorded; the spread EMA is only trusted after enough observations have accumulated

is_closed — flag set to 1 when the GAMMA API confirms the market has resolved; trades in closed markets are skipped in the trading loop

last_checked — unix timestamp of the last spread update


# pnl_history

id — autoincrement primary key

timestamp — unix timestamp of the snapshot

total_investment — sum of buy_price * shares_bought across all OPEN trades

total_pnl — combined realized and unrealized PNL

realized_pnl — PNL from CLOSED trades only

unrealized_pnl — PNL from currently OPEN trades

avg_pnl_per_share — mean PNL per share across all positions, a size-normalized performance metric

roi — total_pnl / total_investment

num_open_trades / num_closed_trades — counts at snapshot time

avg_edge — mean buy_edge across open positions, giving a sense of whether the portfolio is drifting toward lower-conviction entries

total_shares — total shares held across all open positions
