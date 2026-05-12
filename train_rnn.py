import sqlite3
import json
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import random

# this file is for training a recurrent neural network used in RNN_trade_bot.py that evaluates trade market data to determine expected value

DB_FILE = "trades.db"
MODEL_PATH = "new_rnn_history_model.pt"

def load_training_rows(db_file):
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            condition_id,
            outcome_idx,
            market_title,
            outcome_label,
            status,
            current_price,
            history,
            buy_time
        FROM trades
        WHERE status IN ('WATCH', 'WATCHED')
          AND current_price IS NOT NULL
          AND history IS NOT NULL
    """).fetchall()

    conn.close()

    out = []
    for r in rows:
        try:
            hist = json.loads(r["history"])
        except Exception:
            continue

        if not isinstance(hist, list) or len(hist) < 50:
            continue

        try:
            cp = float(r["current_price"])
        except Exception:
            continue

        # map near-0/near-1 to binary outcome; skip ambiguous prices
        if cp <= 0.01:
            resolved = 0
        elif cp >= 0.99:
            resolved = 1
        else:
            continue

        out.append({
            "market_id": r["condition_id"],
            "market_title": r["market_title"],
            "resolved_outcome": resolved,
            "history": hist,
            "buy_time": r["buy_time"]
        })

    return out

def build_step_features(row, first_ts, first_price, market_title):
    if not isinstance(row, list) or len(row) < 6:
        return None

    try:
        bid_price, ask_price, edge, w_for, w_against, ts = row[:6]
        bid_price = float(bid_price)
        ask_price = float(ask_price) if ask_price is not None else None
        mid_price = (bid_price + ask_price) / 2 if ask_price is not None else bid_price
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
        title = (market_title or "").lower()
        esport = 1 if ("lol" in title or "dota 2" in title or "counter-strike" in title) else 0

    bid_logit = logit(bid_price)
    ask_logit = logit(ask_price) if ask_price is not None else bid_logit
    mid_logit = logit(mid_price)
    first_price_logit = logit(first_price)
    avg_price_for_logit = logit(avg_price_for) if avg_price_for is not None else 0.0
    avg_price_against_logit = logit(avg_price_against) if avg_price_against is not None else 0.0

    delta_price_for = (mid_price - float(avg_price_for)) if avg_price_for is not None else 0.0
    delta_price_against = (mid_price - float(avg_price_against)) if avg_price_against is not None else 0.0

    feats = [
        bid_logit,
        ask_logit,
        mid_logit,
        first_price_logit,
        avg_price_for_logit,
        avg_price_against_logit,
        logit(min(max(delta_price_for + 0.5, 0.0), 1.0)),
        logit(min(max(delta_price_against + 0.5, 0.0), 1.0)),
        logit(min(max(mid_price - first_price + 0.5, 0.0), 1.0)),
        edge,
        edge / max(mid_price, 0.01),
        w_for,
        w_for - w_against,
        ts - first_ts,                     # dt
        first_price,                       # first_price (mid)
        mid_price - first_price,           # price_change_from_first
        float(n_sharps_for) if n_sharps_for is not None else 0.0,
        float(n_sharps_against) if n_sharps_against is not None else 0.0,
        float(avg_sharp_for) if avg_sharp_for is not None else 0.0,
        float(avg_sharp_against) if avg_sharp_against is not None else 0.0,
        delta_price_for,
        delta_price_against,
    ]

    return feats


def build_sequences(markets, cutoff_low=0.01, cutoff_high=0.99):
    sequences = []
    labels = []
    groups = []
    price_sequences = []
    bid_sequences = []
    ask_sequences = []
    kept_indices = []

    for mi, m in enumerate(markets):
        hist = m["history"]
        if not hist:
            continue

        if len(hist) < 50:
            continue

        first_ts = None
        first_price = None
        seq = []
        price_seq = []
        bid_seq = []
        ask_seq = []

        # build forward until cutoff (exclude extreme endpoint)
        for row in hist:
            if not isinstance(row, list) or len(row) < 6:
                continue
            bid_price = float(row[0])
            ask_price = float(row[1]) if row[1] is not None else None
            price_val = (bid_price + ask_price) / 2 if ask_price is not None else bid_price

            if first_ts is None:
                first_ts = row[5]
                first_price = price_val

            if price_val <= cutoff_low or price_val >= cutoff_high:
                break

            feats = build_step_features(row, first_ts, first_price, m.get("market_title"))
            if feats is None:
                continue

            seq.append(feats)
            price_seq.append(price_val)
            bid_seq.append(bid_price)
            ask_seq.append(ask_price)

        if len(seq) < 110:
            continue

        # ensure there exists a valid 100-step subsequence with +10-step future price
        valid_ends = [
            i for i, p in enumerate(price_seq)
            if i >= 99 and (i + 10) < len(price_seq) and 0.05 <= p <= 0.95
        ]
        if not valid_ends:
            continue

        sequences.append(np.array(seq, dtype=np.float32))
        price_sequences.append(price_seq)
        bid_sequences.append(bid_seq)
        ask_sequences.append(ask_seq)
        labels.append(m["resolved_outcome"])
        groups.append(m["market_id"])
        kept_indices.append(mi)
    # labels are still resolved outcomes, but training targets are future prices

    return (
        sequences,
        price_sequences,
        bid_sequences,
        ask_sequences,
        np.array(labels, dtype=np.int64),
        np.array(groups),
        np.array(kept_indices, dtype=np.int64),
    )


class SeqDataset(Dataset):
    def __init__(
        self,
        sequences,
        price_sequences,
        bid_sequences,
        ask_sequences,
        labels,
        return_index=False,
        subseq_len=100,
        price_min=0.05,
        price_max=0.95
    ):
        self.sequences = sequences
        self.price_sequences = price_sequences
        self.bid_sequences = bid_sequences
        self.ask_sequences = ask_sequences
        self.labels = labels
        self.return_index = return_index
        self.subseq_len = subseq_len
        self.price_min = price_min
        self.price_max = price_max
        self.valid_end_indices = []
        min_end = subseq_len - 1
        for price_seq in price_sequences:
            valid = [
                i for i, p in enumerate(price_seq)
                if i >= min_end and price_min <= p <= price_max
            ]
            self.valid_end_indices.append(valid)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.return_index:
            return (
                self.sequences[idx],
                self.price_sequences[idx],
                self.bid_sequences[idx],
                self.ask_sequences[idx],
                self.valid_end_indices[idx],
                self.labels[idx],
                idx
            )
        return (
            self.sequences[idx],
            self.price_sequences[idx],
            self.bid_sequences[idx],
            self.ask_sequences[idx],
            self.valid_end_indices[idx],
            self.labels[idx]
        )


def collate_batch(batch):
    if len(batch[0]) == 7:
        sequences, price_sequences, bid_sequences, ask_sequences, valid_ends, labels, indices = zip(*batch)
        idx = torch.tensor(indices, dtype=torch.long)
    else:
        sequences, price_sequences, bid_sequences, ask_sequences, valid_ends, labels = zip(*batch)
        idx = None
    y = torch.tensor(labels, dtype=torch.float32)
    return sequences, price_sequences, bid_sequences, ask_sequences, valid_ends, y, idx

horizon = 3
def sample_subseq_batch(
    sequences,
    price_sequences,
    bid_sequences,
    ask_sequences,
    valid_ends,
    labels,
    subseq_len=100,
    horizon=horizon,
    device=None
):
    xs = []
    ys = []
    market_prices = []
    bid_prices = []
    ask_prices = []
    bid_prices_future = []
    ask_prices_future = []
    for seq, price_seq, bid_seq, ask_seq, ends, y in zip(sequences, price_sequences, bid_sequences, ask_sequences, valid_ends, labels):
        if not ends:
            continue
        end = random.choice(ends)
        start = end - subseq_len + 1
        if start < 0:
            continue
        sub = seq[start:end + 1]
        if len(sub) != subseq_len:
            continue
        if (end + horizon) >= len(price_seq):
            continue
        xs.append(torch.from_numpy(sub))
        ys.append(logit(price_seq[end + horizon]))
        # price_seq already stores mid prices; clamp for safety
        market_prices.append(min(max(float(price_seq[end]), 0.0), 1.0))
        bid_prices.append(bid_seq[end])
        ask_prices.append(ask_seq[end])
        bid_prices_future.append(bid_seq[end + horizon])
        ask_prices_future.append(ask_seq[end + horizon])
    if not xs:
        return None
    x = torch.stack(xs, dim=0)
    lengths = torch.full((x.size(0),), subseq_len, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.float32)
    if device is not None:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)
    return x, lengths, y, market_prices, bid_prices, ask_prices, bid_prices_future, ask_prices_future


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        last = h_n[-1]
        logits = self.head_out(last).squeeze(-1)
        return logits


def logit(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def bce_loss(logits, targets):
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def mse_loss(preds, targets):
    return nn.functional.mse_loss(preds, targets)


def standardize_sequences(sequences, mean=None, std=None):
    all_feats = np.concatenate(sequences, axis=0)
    if mean is None or std is None:
        mean = all_feats.mean(axis=0)
        std = all_feats.std(axis=0) + 1e-8
    normed = [(seq - mean) / std for seq in sequences]
    return normed, mean, std


def main():
    markets = load_training_rows(DB_FILE)
    print(f"Loaded {len(markets)} markets")
    if not markets:
        return

    sequences, price_sequences, bid_sequences, ask_sequences, labels, groups, kept_indices = build_sequences(
        markets, cutoff_low=0.01, cutoff_high=0.99
    )
    print(f"Sequences: {len(sequences)}")
    if len(sequences) < 10:
        print("Not enough sequences.")
        return

    # time-based split (by buy_time)
    times = np.array([m.get("buy_time") or 0 for m in markets], dtype=np.float64)
    kept_times = times[kept_indices]
    order = np.argsort(kept_times)
    cutoff = int(len(order) * 0.85)
    train_idx = order[:cutoff]
    val_idx = order[cutoff:]

    train_seqs_raw = [sequences[i] for i in train_idx]
    train_price_seqs = [price_sequences[i] for i in train_idx]
    train_bid_seqs = [bid_sequences[i] for i in train_idx]
    train_ask_seqs = [ask_sequences[i] for i in train_idx]
    val_seqs_raw = [sequences[i] for i in val_idx]
    val_price_seqs_raw = [price_sequences[i] for i in val_idx]
    val_bid_seqs = [bid_sequences[i] for i in val_idx]
    val_ask_seqs = [ask_sequences[i] for i in val_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    # sanity check: no overlap between train/val markets
    train_ids = {markets[kept_indices[i]]["market_id"] for i in train_idx}
    val_ids = {markets[kept_indices[i]]["market_id"] for i in val_idx}
    overlap_ids = train_ids & val_ids
    if overlap_ids:
        print(f"Overlap in condition_id between train/val: {len(overlap_ids)}")

    # standardize using training data only 
    all_train_feats = np.concatenate(train_seqs_raw, axis=0)
    mean = all_train_feats.mean(axis=0)
    std = all_train_feats.std(axis=0) + 1e-8
    train_seqs, _, _ = standardize_sequences(train_seqs_raw, mean, std)
    val_seqs, _, _ = standardize_sequences(val_seqs_raw, mean, std)

    train_ds = SeqDataset(
        train_seqs,
        train_price_seqs,
        train_bid_seqs,
        train_ask_seqs,
        y_train,
        return_index=True,
        subseq_len=100,
        price_min=0.05,
        price_max=0.95
    )
    val_ds = SeqDataset(
        val_seqs,
        val_price_seqs_raw,
        val_bid_seqs,
        val_ask_seqs,
        y_val,
        return_index=True,
        subseq_len=100,
        price_min=0.05,
        price_max=0.95
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_batch)

    input_dim = train_seqs[0].shape[1]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    hidden_dim = 64
    num_layers = 1
    dropout = 0.1
    bidirectional = False
    model = RNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=1e-6
    )

    best_val = float("inf")
    patience = 100
    patience_left = patience

    for epoch in range(1, 1000):
        model.train()
        total_loss = 0.0
        for sequences, price_sequences, bid_sequences, ask_sequences, valid_ends, y, _ in train_loader:
            batch = sample_subseq_batch(
                sequences,
                price_sequences,
                bid_sequences,
                ask_sequences,
                valid_ends,
                y,
                subseq_len=100,
                device=device
            )
            if batch is None:
                continue
            x, lengths, y, _, _, _, _, _ = batch
            logits = model(x, lengths)
            loss = mse_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_probs = []
        all_y = []
        market_prices = []
        all_bid_prices = []
        all_ask_prices = []
        all_bid_prices_future = []
        all_ask_prices_future = []
        pnl_total = 0.0
        investment_total = 0.0
        compute_pnl = False
        with torch.inference_mode():
            for sequences, price_sequences, bid_sequences, ask_sequences, valid_ends, y, _ in val_loader:
                # build up to 5 subsequences per market and average predictions/targets
                sub_x = []
                sub_item_idx = []
                sub_entry_mid = []
                sub_entry_bid = []
                sub_entry_ask = []
                sub_exit_bid = []
                sub_exit_ask = []
                sub_y = []
                sampled_ends_map = {}
                for i, (seq, price_seq, bid_seq, ask_seq, ends) in enumerate(
                    zip(sequences, price_sequences, bid_sequences, ask_sequences, valid_ends)
                ):
                    if not ends:
                        continue
                    k = min(2, len(ends))
                    sampled_ends = random.sample(ends, k)
                    sampled_ends_map[i] = sampled_ends
                    for end in sampled_ends:
                        start = end - 100 + 1
                        if start < 0 or (end + horizon) >= len(price_seq):
                            continue
                        sub = seq[start:end + 1]
                        if len(sub) != 100:
                            continue
                        sub_x.append(torch.from_numpy(sub))
                        sub_item_idx.append(i)
                        entry_mid = min(max(float(price_seq[end]), 0.0), 1.0)
                        sub_entry_mid.append(entry_mid)
                        sub_entry_bid.append(bid_seq[end])
                        sub_entry_ask.append(ask_seq[end])
                        sub_exit_bid.append(bid_seq[end + horizon])
                        sub_exit_ask.append(ask_seq[end + horizon])
                        sub_y.append(logit(price_seq[end + horizon]))

                if not sub_x:
                    continue

                x = torch.stack(sub_x, dim=0).to(device)
                lengths = torch.full((x.size(0),), 100, dtype=torch.long, device=device)
                y_sub = torch.tensor(sub_y, dtype=torch.float32, device=device)

                logits = model(x, lengths)
                probs = torch.sigmoid(logits)
                loss = mse_loss(logits, y_sub)
                val_loss += loss.item()
                val_batches += 1

                # average per original item
                item_count = {}
                item_prob_sum = {}
                item_y_sum = {}
                item_entry_mid_sum = {}
                item_entry_bid_sum = {}
                item_entry_ask_sum = {}
                item_exit_bid_sum = {}
                item_exit_ask_sum = {}

                probs_cpu = probs.cpu().numpy().tolist()
                y_cpu = torch.sigmoid(y_sub).cpu().numpy().tolist()
                for idx, p, yy, em, eb, ea, xb, xa in zip(
                    sub_item_idx,
                    probs_cpu,
                    y_cpu,
                    sub_entry_mid,
                    sub_entry_bid,
                    sub_entry_ask,
                    sub_exit_bid,
                    sub_exit_ask,
                ):
                    item_count[idx] = item_count.get(idx, 0) + 1
                    item_prob_sum[idx] = item_prob_sum.get(idx, 0.0) + p
                    item_y_sum[idx] = item_y_sum.get(idx, 0.0) + yy
                    item_entry_mid_sum[idx] = item_entry_mid_sum.get(idx, 0.0) + em
                    item_entry_bid_sum[idx] = item_entry_bid_sum.get(idx, 0.0) + (eb if eb is not None else em)
                    item_entry_ask_sum[idx] = item_entry_ask_sum.get(idx, 0.0) + (ea if ea is not None else em)
                    item_exit_bid_sum[idx] = item_exit_bid_sum.get(idx, 0.0) + (xb if xb is not None else yy)
                    item_exit_ask_sum[idx] = item_exit_ask_sum.get(idx, 0.0) + (xa if xa is not None else yy)

                for idx, cnt in item_count.items():
                    all_probs.append(item_prob_sum[idx] / cnt)
                    all_y.append(item_y_sum[idx] / cnt)
                    market_prices.append(item_entry_mid_sum[idx] / cnt)
                    all_bid_prices.append(item_entry_bid_sum[idx] / cnt)
                    all_ask_prices.append(item_entry_ask_sum[idx] / cnt)
                    all_bid_prices_future.append(item_exit_bid_sum[idx] / cnt)
                    all_ask_prices_future.append(item_exit_ask_sum[idx] / cnt)

                if compute_pnl:
                    # PnL simulation: hold while model EV positive at t+horizon (batched)
                    min_edge = 0.01
                    # precompute predictions for all sampled ends and their horizon steps
                    pred_cache = {}
                    batch_subs = []
                    batch_keys = []
                    for i, (seq, price_seq, bid_seq, ask_seq) in enumerate(
                        zip(sequences, price_sequences, bid_sequences, ask_sequences)
                    ):
                        ends = sampled_ends_map.get(i, [])
                        for end in ends:
                            start = end - 100 + 1
                            if start < 0 or (end + horizon) >= len(price_seq):
                                continue
                            sub = seq[start:end + 1]
                            if len(sub) != 100:
                                continue
                            batch_subs.append(torch.from_numpy(sub))
                            batch_keys.append((i, end))
                    if batch_subs:
                        bx = torch.stack(batch_subs, dim=0).to(device)
                        bl = torch.full((bx.size(0),), 100, dtype=torch.long, device=device)
                        bprobs = torch.sigmoid(model(bx, bl)).cpu().numpy().tolist()
                        for key, pval in zip(batch_keys, bprobs):
                            pred_cache[key] = pval

                    for i, (seq, price_seq, bid_seq, ask_seq) in enumerate(
                        zip(sequences, price_sequences, bid_sequences, ask_sequences)
                    ):
                        ends = sampled_ends_map.get(i, [])
                        if not ends:
                            continue
                        for end in ends:
                            p_entry = pred_cache.get((i, end))
                            if p_entry is None:
                                continue
                            entry_mid = price_seq[end]
                            entry_bid = bid_seq[end] if bid_seq[end] is not None else entry_mid
                            entry_ask = ask_seq[end] if ask_seq[end] is not None else entry_mid
                            if entry_mid is None:
                                continue

                            direction = None
                            if p_entry > entry_mid + min_edge:
                                direction = "YES"
                                entry_cost = entry_ask
                            elif p_entry < entry_mid - min_edge:
                                direction = "NO"
                                entry_cost = 1 - entry_bid
                            if direction is None:
                                continue

                            exit_idx = end + horizon
                            while exit_idx < len(price_seq):
                                if exit_idx - 99 < 0:
                                    break
                                p_next = pred_cache.get((i, exit_idx))
                                if p_next is None:
                                    sub = seq[exit_idx - 99:exit_idx + 1]
                                    if len(sub) != 100:
                                        break
                                    x_next = torch.from_numpy(sub).unsqueeze(0).to(device)
                                    lengths_next = torch.tensor([sub.shape[0]], dtype=torch.long, device=device)
                                    p_next = torch.sigmoid(model(x_next, lengths_next)).item()
                                    pred_cache[(i, exit_idx)] = p_next

                                mid_next = price_seq[exit_idx]
                                if mid_next is None:
                                    break
                                ev_next = (p_next - mid_next) if direction == "YES" else (mid_next - p_next)
                                if ev_next > 0 and (exit_idx + horizon) < len(price_seq):
                                    exit_idx += horizon
                                    continue
                                break

                            if exit_idx >= len(price_seq):
                                exit_idx = len(price_seq) - 1
                            exit_mid = price_seq[exit_idx]
                            exit_bid = bid_seq[exit_idx] if bid_seq[exit_idx] is not None else exit_mid
                            exit_ask = ask_seq[exit_idx] if ask_seq[exit_idx] is not None else exit_mid
                            if direction == "YES":
                                investment_total += entry_cost
                                pnl_total += exit_bid - entry_cost
                            else:
                                investment_total += entry_cost
                                pnl_total += (1 - exit_ask) - entry_cost

        val_loss /= max(1, val_batches)
        mse = float(np.mean((np.array(all_probs) - np.array(all_y)) ** 2)) if all_probs else 0.0
        mae = float(np.mean(np.abs(np.array(all_probs) - np.array(all_y)))) if all_probs else 0.0
        brier_market = float(np.mean((np.array(market_prices) - np.array(all_y)) ** 2)) if market_prices else 0.0

        roi = (pnl_total / investment_total * 100) if investment_total > 0 else 0.0

        print(f"Epoch {epoch:02d} | train_loss={total_loss/len(train_loader):.4f} | val_loss={val_loss:.4f} | MSE={mse:.4f} | MAE={mae:.4f} | Market MSE={brier_market:.4f} | {round(brier_market - mse, 4)} | PnL={pnl_total:.2f} | ROI={roi:.2f}")

        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
            torch.save({
                "model_state": model.state_dict(),
                "mean": mean,
                "std": std,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "bidirectional": bidirectional
            }, MODEL_PATH)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # PnL simulation only once using best model
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        pnl_total = 0.0
        investment_total = 0.0
        all_probs = []
        all_y = []
        market_prices = []
        all_bid_prices = []
        all_ask_prices = []
        all_bid_prices_future = []
        all_ask_prices_future = []
        val_loss = 0.0
        val_batches = 0
        with torch.inference_mode():
            for sequences, price_sequences, bid_sequences, ask_sequences, valid_ends, y, _ in val_loader:
                sub_x = []
                sub_item_idx = []
                sub_entry_mid = []
                sub_entry_bid = []
                sub_entry_ask = []
                sub_exit_bid = []
                sub_exit_ask = []
                sub_y = []
                sampled_ends_map = {}
                for i, (seq, price_seq, bid_seq, ask_seq, ends) in enumerate(
                    zip(sequences, price_sequences, bid_sequences, ask_sequences, valid_ends)
                ):
                    if not ends:
                        continue
                    k = min(2, len(ends))
                    sampled_ends = random.sample(ends, k)
                    sampled_ends_map[i] = sampled_ends
                    for end in sampled_ends:
                        start = end - 100 + 1
                        if start < 0 or (end + horizon) >= len(price_seq):
                            continue
                        sub = seq[start:end + 1]
                        if len(sub) != 100:
                            continue
                        sub_x.append(torch.from_numpy(sub))
                        sub_item_idx.append(i)
                        entry_mid = min(max(float(price_seq[end]), 0.0), 1.0)
                        sub_entry_mid.append(entry_mid)
                        sub_entry_bid.append(bid_seq[end])
                        sub_entry_ask.append(ask_seq[end])
                        sub_exit_bid.append(bid_seq[end + horizon])
                        sub_exit_ask.append(ask_seq[end + horizon])
                        sub_y.append(logit(price_seq[end + horizon]))

                if not sub_x:
                    continue

                x = torch.stack(sub_x, dim=0).to(device)
                lengths = torch.full((x.size(0),), 100, dtype=torch.long, device=device)
                y_sub = torch.tensor(sub_y, dtype=torch.float32, device=device)

                logits = model(x, lengths)
                probs = torch.sigmoid(logits)
                loss = mse_loss(logits, y_sub)
                val_loss += loss.item()
                val_batches += 1

                item_count = {}
                item_prob_sum = {}
                item_y_sum = {}
                item_entry_mid_sum = {}
                item_entry_bid_sum = {}
                item_entry_ask_sum = {}
                item_exit_bid_sum = {}
                item_exit_ask_sum = {}

                probs_cpu = probs.cpu().numpy().tolist()
                y_cpu = torch.sigmoid(y_sub).cpu().numpy().tolist()
                for idx, p, yy, em, eb, ea, xb, xa in zip(
                    sub_item_idx,
                    probs_cpu,
                    y_cpu,
                    sub_entry_mid,
                    sub_entry_bid,
                    sub_entry_ask,
                    sub_exit_bid,
                    sub_exit_ask,
                ):
                    item_count[idx] = item_count.get(idx, 0) + 1
                    item_prob_sum[idx] = item_prob_sum.get(idx, 0.0) + p
                    item_y_sum[idx] = item_y_sum.get(idx, 0.0) + yy
                    item_entry_mid_sum[idx] = item_entry_mid_sum.get(idx, 0.0) + em
                    item_entry_bid_sum[idx] = item_entry_bid_sum.get(idx, 0.0) + (eb if eb is not None else em)
                    item_entry_ask_sum[idx] = item_entry_ask_sum.get(idx, 0.0) + (ea if ea is not None else em)
                    item_exit_bid_sum[idx] = item_exit_bid_sum.get(idx, 0.0) + (xb if xb is not None else yy)
                    item_exit_ask_sum[idx] = item_exit_ask_sum.get(idx, 0.0) + (xa if xa is not None else yy)

                for idx, cnt in item_count.items():
                    all_probs.append(item_prob_sum[idx] / cnt)
                    all_y.append(item_y_sum[idx] / cnt)
                    market_prices.append(item_entry_mid_sum[idx] / cnt)
                    all_bid_prices.append(item_entry_bid_sum[idx] / cnt)
                    all_ask_prices.append(item_entry_ask_sum[idx] / cnt)
                    all_bid_prices_future.append(item_exit_bid_sum[idx] / cnt)
                    all_ask_prices_future.append(item_exit_ask_sum[idx] / cnt)

                # PnL simulation: hold while model EV positive at t+horizon (batched)
                min_edge = 0.01
                pred_cache = {}
                batch_subs = []
                batch_keys = []
                for i, (seq, price_seq, bid_seq, ask_seq) in enumerate(
                    zip(sequences, price_sequences, bid_sequences, ask_sequences)
                ):
                    ends = sampled_ends_map.get(i, [])
                    for end in ends:
                        start = end - 100 + 1
                        if start < 0 or (end + horizon) >= len(price_seq):
                            continue
                        sub = seq[start:end + 1]
                        if len(sub) != 100:
                            continue
                        batch_subs.append(torch.from_numpy(sub))
                        batch_keys.append((i, end))
                if batch_subs:
                    bx = torch.stack(batch_subs, dim=0).to(device)
                    bl = torch.full((bx.size(0),), 100, dtype=torch.long, device=device)
                    bprobs = torch.sigmoid(model(bx, bl)).cpu().numpy().tolist()
                    for key, pval in zip(batch_keys, bprobs):
                        pred_cache[key] = pval

                for i, (seq, price_seq, bid_seq, ask_seq) in enumerate(
                    zip(sequences, price_sequences, bid_sequences, ask_sequences)
                ):
                    ends = sampled_ends_map.get(i, [])
                    if not ends:
                        continue
                    for end in ends:
                        p_entry = pred_cache.get((i, end))
                        if p_entry is None:
                            continue
                        entry_mid = price_seq[end]
                        entry_bid = bid_seq[end] if bid_seq[end] is not None else entry_mid
                        entry_ask = ask_seq[end] if ask_seq[end] is not None else entry_mid
                        if entry_mid is None:
                            continue

                        direction = None
                        if p_entry > entry_mid + min_edge:
                            direction = "YES"
                            entry_cost = entry_ask
                        elif p_entry < entry_mid - min_edge:
                            direction = "NO"
                            entry_cost = 1 - entry_bid
                        if direction is None:
                            continue

                        exit_idx = end + horizon
                        while exit_idx < len(price_seq):
                            if exit_idx - 99 < 0:
                                break
                            p_next = pred_cache.get((i, exit_idx))
                            if p_next is None:
                                sub = seq[exit_idx - 99:exit_idx + 1]
                                if len(sub) != 100:
                                    break
                                x_next = torch.from_numpy(sub).unsqueeze(0).to(device)
                                lengths_next = torch.tensor([sub.shape[0]], dtype=torch.long, device=device)
                                p_next = torch.sigmoid(model(x_next, lengths_next)).item()
                                pred_cache[(i, exit_idx)] = p_next

                            mid_next = price_seq[exit_idx]
                            if mid_next is None:
                                break
                            ev_next = (p_next - mid_next) if direction == "YES" else (mid_next - p_next)
                            if ev_next > 0 and (exit_idx + horizon) < len(price_seq):
                                exit_idx += horizon
                                continue
                            break

                        if exit_idx >= len(price_seq):
                            exit_idx = len(price_seq) - 1
                        exit_mid = price_seq[exit_idx]
                        exit_bid = bid_seq[exit_idx] if bid_seq[exit_idx] is not None else exit_mid
                        exit_ask = ask_seq[exit_idx] if ask_seq[exit_idx] is not None else exit_mid
                        if direction == "YES":
                            investment_total += entry_cost
                            pnl_total += exit_bid - entry_cost
                        else:
                            investment_total += entry_cost
                            pnl_total += (1 - exit_ask) - entry_cost

        val_loss = val_loss / max(1, val_batches)
        mse = float(np.mean((np.array(all_probs) - np.array(all_y)) ** 2)) if all_probs else 0.0
        mae = float(np.mean(np.abs(np.array(all_probs) - np.array(all_y)))) if all_probs else 0.0
        brier_market = float(np.mean((np.array(market_prices) - np.array(all_y)) ** 2)) if market_prices else 0.0
        roi = (pnl_total / investment_total * 100) if investment_total > 0 else 0.0
        print(f"Best model eval | val_loss={val_loss:.4f} | MSE={mse:.4f} | MAE={mae:.4f} | Market MSE={brier_market:.4f} | {round(brier_market - mse, 4)} | PnL={pnl_total:.2f} | ROI={roi:.2f}")
    except Exception as e:
        print(f"Best model eval failed: {e}")

    print(f"Saved best model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
