# SPY42 v1: dataset and training audit — direction-only redesign

**Status:** Model v1 was trained and published; it has not shown a meaningful directional edge. **No v2 retraining or 2026 final-test evaluation has been started.** The exact target definition needs to be agreed before starting v2.

## What each chart contains

One training example is a `42` completed **SPY daily candles** chart, with real OHLC candles, EMA8, EMA21 and volume. The original renderer produces a 1920×1080 PNG; training crops the price/EMA and volume panels, stacks them, and resizes to **768×336 RGB pixels**. The model receives pixels only. The input chart ends *before* the next-day label date.

Data source in the notebook's verified embedded bundle: `SPY_daily_with_2019_warmup.csv`, fixed historical snapshot through 2026-11-20; 2019 warms up EMAs and ATR.

| Group | Period by target date | Chart images | Non-doji direction labels |
|---|---|---:|---:|
| Train | 2020–2024 | 1,258 | 1,256 |
| Validation/development | 2025 | 250 | 249 |
| Locked final test | 2026 through Nov 20 | 224 | Not evaluated |
| Latest input | As-of Nov 20, 2026 | 1 | No label |

Adjacent charts share 41 of 42 candles, so 1,258 examples are not 1,258 independent market events.

## Precisely what labels v1 used

```python
next_day_body = next_day_close - next_day_open
target_up = 1 if next_day_body > 0 else (0 if next_day_body < 0 else None)
target_body_atr = next_day_body / ATR14_known_on_chart_date
```

The first label predicts **green/red for the *next day's open-to-close candle body***, NOT whether the next close is above the previous close, not 3–5-day bearish trend, and not an EMA-triggered trade. The second label, the body size in ATR units, was also learned despite the user's direction-only goal.

The underlying manifest includes diagnostics such as `ema8_touched` and `ema21_touched`, but **these are not training labels**; no manual trade entries, stops, conditional setups, five-minute candles or options profits were used for training.

**2022 direction mismatch examples:**

- 2022-01-10: prior close $466.09; next open $462.70; next close $465.51. V1 labels **GREEN** even though the session closed **below the previous close**.
- 2022-01-12: prior close $469.75; next open $471.59; next close $471.02. V1 labels **RED** even though it closed **above the previous close**.
- Exactly **41/250 (16.4%)** of comparable 2022 sessions disagree between these two reasonable definitions of 'UP'.

## The bear-market data was present

| Year | Green daily bodies | Red daily bodies | Exact doji |
|---|---:|---:|---:|
| 2020 | 147 | 105 | 1 |
| 2021 | 137 | 114 | 1 |
| 2022 | 127 | 124 | 0 |
| 2023 | 147 | 103 | 0 |
| 2024 | 135 | 117 | 0 |
| **Total training** | **693** | **563** | **2** |

During the 2020-02-20 to 2020-03-23 COVID crash window the data show **14 red versus 9 green daily bodies**. Despite an approximately 20% decline over calendar 2022, 50.6% of 2022 daily bodies were green. A bearish trend is not a red-candle label.

## What really trained in v1

```text
768×336 RGB chart image
  → pretrained ImageNet ResNet18 backbone: 11,176,512 FROZEN parameters
  → 512 image features → dropout 0.15
      → linear direction head: 513 trainable parameters
      → linear candle-body quantile head: 1,539 trainable parameters
```

Only **2,052 parameters (0.018% of 11,178,564 total)** were optimized. The backbone was not adapted to SPY charts. Thus v1 was real image-based training, but not substantive visual-network fine-tuning.

Loss: **binary cross entropy of next-day green/red + 0.25 × 3-quantile pinball candle-body loss**. Optimizer: AdamW, LR=0.001, batch=16, cosine schedule, max 15 epochs, early stopping patience 4. CPU training stopped after eight epochs (~21 minutes), selecting **epoch 4 on a combined direction+body validation objective**.

## Actual 2025 validation

- Direction: **132/249 = 53.01%** accurate.
- Always-green majority baseline: **131/249 = 52.61%** (only one fewer correct).
- Calibrated direction log loss **0.68953** vs approximately **0.69315** for constant 0.5.
- The selected signal threshold produced **74 CALL_BIAS and 0 PUT_BIAS**.
- Nominal 80% body interval covered **60.4%**.
- Illustrative SPY signed-return proxy assumed zero costs and is **not** executable options-trading performance.

Source: [original training run](https://github.com/jamelof23/SPY/actions/runs/45091374048) and [detailed 2025 report](https://github.com/jamelof23/SPY/blob/main/reports/SPY42_2025_VALIDATION.md).

## Mandatory decisions before v2

**Choose the intended direction label**:

1. **Next-day candle body:** next close versus *next open* (v1 target).
2. **Next-day price direction:** next close versus *previous close*, including overnight gaps.
3. **3–5-day trend:** close at a predeclared multi-day horizon versus current close.
4. **Expert CALL/PUT/WAIT trade decision:** requires labeled examples of the user's actual trading setup and, for five-minute EMA entries, corresponding intraday observations.

Rebuilding the network before confirming the correct label risks repeating the same mistake.

**Proposed direction-only v2 after label confirmation:**
- Remove the candle-body head and quantile loss. One direction output; consider an explicit abstention threshold for WAIT.
- Primary model: a **small, fully trainable CNN initialized from scratch** specifically for candlestick/EMA/volume chart pixels. Compare a partially unfrozen ResNet only as a clearly identified alternative.
- Keep chronological source splits, avoid time-reversal/horizontal flips, and retain the 2026 test locked.
- Use walk-forward development folds within 2020–2024 to compare architectures and label-balanced training. Use 2025 for the pre-2026 validation report and disclose that 2025 was also used in v1.
- Publish all per-class metrics (UP and DOWN precision/recall, balanced accuracy, confusion matrix), signal coverage and conditional signal accuracy, comparison against trivial/EMA baselines, explicit training code/number of trainable parameters, and loss curves. An 80% hit rate on selected expert setups is not identical to 80% accuracy on every day.
- Freeze the resulting design, model, calibration, and trading-cost assumptions **before** evaluating the 224 held-out 2026 targets once.

This report is a reproducible audit/specification. No 2026 outcomes were read for it.
