# SPY42 v2 · five-session direction labels and training readiness

**Status:** Labels designed and audited; v2 neural-network training has **not** started; 2026 test outcomes have **not** been evaluated.

## Primary supervised target

At the close of trading session `t`, use its last **42 completed SPY daily candles**, EMA8, EMA21 and volume to forecast where SPY closes **five trading sessions later**.

- Label `UP=1` iff `Close[t+5] > Close[t]`.
- Label `DOWN=0` iff `Close[t+5] < Close[t]`.
- Exact ties excluded (predeclared).
- **No candle-body magnitude target and no next-day green/red candle target.**
- The future 5-session close is a **training label only**; it is never an input feature.

Existing 42-candle chart PNGs are reused, with model inputs limited to image pixels.

## Audited source and counts

Source: fixed `SPY_daily_with_2019_warmup.csv`, SHA256 `4e2045b50251b87619b406abcdee3697862a2d69cc18744075fc15683ab57c5d`. A strictly pre-2026 builder excludes all rows dated 2026 or later **before** reading prices or calculating indicators.

| Chronological group | Date range | Labeled 42-chart examples | UP | DOWN | Always-UP baseline |
|---|---|---:|---:|---:|---:|
| Training | 2020–2024 | 1,253 | 752 | 501 | 60.02% |
| 2025 development validation | 2025 | 245 | 155 | 90 | 63.27% |
| 2026 final test | 2026 | **Not loaded** | — | — | **Untouched** |

Five trading sessions crossing a train/development year boundary are purged rather than mislabeled. Adjacent windows and forecast targets overlap; do not treat adjacent examples as independent observations.

## Bear-market countertrend rebounds and EMA failures

We also tag a **contemporaneously observable setup** for separate accuracy reporting:
- EMA8 below EMA21.
- EMA21 falling versus its value five sessions ago.
- Current closing price below EMA21.
- At least two of last three candles were green **and** current close above the close three sessions ago.
- Current day's observed high-low range comes within 0.5 ATR of EMA8 or EMA21.

This is an **as-of-date contextual tag**, not a future-dependent label and not a discovered executable trade-entry rule. Audited data show **72 training setup-days from about 25 temporally separated episodes**, and **9 such development setup-days in 2025**. Therefore a setup-specific accuracy figure has substantial statistical uncertainty; don't promise 80% hit rate from these limited examples.

A five-day higher/lower closing price is different from detecting precisely which candle rejects an EMA; the latter may eventually require expert annotations and 5-minute data.

## Proposed training comparison

1. Sanity check: always-UP baseline, EMA/trend heuristic and a simple tabular baseline used **only for comparison** (the CNN still receives image pixels only).
2. Main model: **small CNN, all convolutional layers trainable from scratch**, one sigmoid direction head, class weighting considered from training counts only, checkpoint selection using only development data.
3. Alternative: partially fine-tuned ResNet18 **clearly named** and compared against the CNN; no frozen backbones presented as chart-specific training.
4. Measure CPU **one-epoch speed benchmark** before committing to the full training configuration; prefer GPU when available and affordable.
5. Save confusion matrix; UP/DOWN precision and recall; balanced accuracy; Brier/log-loss; calibration; signal coverage; overall vs bear-pullback context performance; training history and exact model parameters.
6. Freeze model/settings/thresholds before the single 2026 final test. Using 2025 for v2 after v1 means it remains a **development** set, not pristine evaluation.

## Compute decision

- Standard GitHub Actions runners in this **public** repository offer 4 CPU cores, no GPU and free public-repo execution; hosted job maximum is 6 hours.
- GPU-powered GitHub larger runners are **paid** and limited to organizations on GitHub Team or Enterprise Cloud; an individual account cannot directly select them for this personal repository. Linux GPU T4 priced **US$0.052/minute** (subject to plan/billing).
- Google Colab Free provides GPU access when available, but may time out/terminate. Do not rely on an unattended Colab session without persistent checkpoints and explicit confirmation of saved results.
- A small trainable CNN might take **1–4 hours on 4-core CPU** or **10–45 minutes on T4 GPU**. A partially fine-tuned ResNet18 may take **2–8 hours on CPU** or **15–60 minutes on T4**. **These are unbenchmarked planning estimates, not measured timing.**
- Measured v1 frozen-ResNet18 (only 2,052 trainable head parameters): 8 CPU epochs in 21 minutes. That timing must not be generalized to full-backbone training.

## Implemented

The causal label-builder script is committed at [`scripts/build_spy42_v2_5day_labels.py`](https://github.com/jamelof23/SPY/blob/main/scripts/build_spy42_v2_5day_labels.py). v2 model training and 2026 test evaluation have **not** been triggered.

Research/runner references:
- [GitHub runner pricing](https://docs.github.com/en/billing/reference/actions-runner-pricing)
- [GitHub larger-runner eligibility](https://docs.github.com/en/actions/concepts/runners/larger-runners)
- [GitHub Actions limits](https://docs.github.com/en/actions/reference/limits)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
