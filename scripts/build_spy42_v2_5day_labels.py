"""Build SPY42 v2 five-session DIRECTION labels using pre-2026 history only.

No model training or 2026 outcome evaluation. Uses existing chart images.
Usage: python scripts/build_spy42_v2_5day_labels.py --source history.csv --output out/
"""
from __future__ import annotations

import argparse
from datetime import date
import csv
import hashlib
import json
import math
from pathlib import Path

HORIZON = 5
TRAIN_YEARS = set(range(2020, 2025))
VALIDATION_YEAR = 2025


def read_pre2026_history(path: Path):
    """Filter out 2026+ rows *before* accessing their numeric values."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        raw = [r for r in csv.DictReader(f) if "2019-01-01" <= r["Date"] < "2026-01-01"]
    raw.sort(key=lambda r: r["Date"])
    assert len({r["Date"] for r in raw}) == len(raw), "Duplicate dates"
    rows, ema8, ema21, atr14 = [], None, None, None
    tr_history = []
    for source in raw:
        d = date.fromisoformat(source["Date"]).isoformat()
        r = {"Date": d}
        for col in ("Open", "High", "Low", "Close"):
            r[col] = float(source[col])
        r["Volume"] = int(source["Volume"])
        o, h, l, c = (r[x] for x in ("Open", "High", "Low", "Close"))
        assert all(math.isfinite(x) and x > 0 for x in (o, h, l, c))
        assert l <= min(o, c) <= max(o, c) <= h and r["Volume"] > 0
        prev_close = rows[-1]["Close"] if rows else c
        tr = max(h-l, abs(h-prev_close), abs(l-prev_close))
        tr_history.append(tr)
        if len(tr_history) == 14:
            atr14 = sum(tr_history) / 14
        elif len(tr_history) > 14:
            atr14 = (13*atr14 + tr) / 14
        ema8 = c if ema8 is None else (2*c + 7*ema8) / 9
        ema21 = c if ema21 is None else (2*c + 20*ema21) / 22
        r.update(EMA8=ema8, EMA21=ema21, ATR14=atr14)
        rows.append(r)
    if not rows or rows[0]["Date"] != "2019-01-02" or rows[-1]["Date"] != "2025-12-31":
        raise ValueError("Expected audited historical source from 2019-01-02 to 2025-12-31")
    return rows


def near_observed_ema_range(r, level, atr_margin=0.5):
    """Distance from known current high-low range to current EMA, in ATR units."""
    distance = max(r["Low"]-level, level-r["High"], 0.0)
    return distance <= atr_margin*r["ATR14"]


def bear_pullback_asof(rows, t):
    """Context tag uses only candles and indicators known at chart end t."""
    r = rows[t]
    bear = (r["EMA8"] < r["EMA21"] and
            r["EMA21"] < rows[t-5]["EMA21"] and
            r["Close"] < r["EMA21"])
    two_green = sum(rows[j]["Close"] > rows[j]["Open"] for j in range(t-2, t+1)) >= 2
    bounce = two_green and r["Close"] > rows[t-3]["Close"]
    near = any(near_observed_ema_range(r, r[name]) for name in ("EMA8", "EMA21"))
    return int(bear), int(bounce), int(near), int(bear and bounce and near)


def produce(source: Path, output: Path):
    rows = read_pre2026_history(source)
    output.mkdir(parents=True, exist_ok=True)
    training, validation, audit = [], [], []
    training_setup_indices = []
    for t in range(41, len(rows)-HORIZON):
        current, end = rows[t], rows[t+HORIZON]
        year, end_year = int(current["Date"][:4]), int(end["Date"][:4])
        if year in TRAIN_YEARS and end_year in TRAIN_YEARS:
            split = "train"
        elif year == VALIDATION_YEAR and end_year == VALIDATION_YEAR:
            split = "validation"
        else:
            continue  # Purge horizons crossing any train/development boundary.
        delta = end["Close"] - current["Close"]
        if delta == 0:
            continue  # Exclude predeclared ties from binary training labels.
        y = int(delta > 0)
        record = dict(
            sample_id="SPY42_" + current["Date"].replace("-", ""),
            asof_date=current["Date"],
            image_path=f'images/{split}/SPY_42_asof_{current["Date"]}.png',
            label_up_5d=y,
        )
        if split == "train":
            training.append(record)
        else:
            validation.append(record)
        bear, bounce, near, setup = bear_pullback_asof(rows, t)
        if setup and split == "train":
            training_setup_indices.append(t)
        audit.append(dict(
            **record, split=split, forecast_end_date=end["Date"],
            asof_close=f'{current["Close"]:.6f}',
            future_5th_close=f'{end["Close"]:.6f}',
            outcome_5d_close_return=f'{end["Close"]/current["Close"]-1:.9f}',
            ema8_asof=f'{current["EMA8"]:.6f}',
            ema21_asof=f'{current["EMA21"]:.6f}',
            atr14_asof=f'{current["ATR14"]:.6f}',
            bear_regime_asof=bear, green_rebound_asof=bounce,
            near_ema_asof=near, bear_pullback_asof=setup,
        ))
    if len(training) != 1253 or len(validation) != 245:
        raise RuntimeError("Unexpected split sizes. Recheck source or calendar.")
    episodes = []
    for i in training_setup_indices:
        if not episodes or i-episodes[-1][-1] > HORIZON:
            episodes.append([i])
        else:
            episodes[-1].append(i)

    def write_csv(name, records):
        with (output/name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write_csv("train_labels.csv", training)
    write_csv("validation_labels.csv", validation)
    write_csv("historical_audit_through_2025.csv", audit)
    counts = lambda rows: {
        "UP": sum(x["label_up_5d"] == 1 for x in rows),
        "DOWN": sum(x["label_up_5d"] == 0 for x in rows),
    }
    summary = dict(
        forecast_horizon_trading_sessions=HORIZON,
        label="UP iff close[t+5]>close[t]; DOWN iff close[t+5]<close[t]; ties excluded",
        input="42 completed SPY daily candles, EMA8, EMA21, volume; image pixels only",
        train=dict(count=len(training), direction_counts=counts(training)),
        validation=dict(count=len(validation), direction_counts=counts(validation)),
        bear_pullback_asof_training_days=len(training_setup_indices),
        bear_pullback_training_episodes_5day_separation=len(episodes),
        bear_pullback_asof_validation_days=sum(
            x["bear_pullback_asof"] for x in audit if x["split"] == "validation"
        ),
        test_2026="NOT loaded, labelled, evaluated or written",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(produce(args.source, args.output), indent=2))
