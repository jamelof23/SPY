# SPY42 · 2025 validation results (model v1)

Source: [completed GitHub Actions validation report](https://github.com/jamelof23/SPY/actions/runs/45096509024) extracted unchanged from the verified saved model release archive. This file summarizes the existing model results; it does not retrain or evaluate 2026.

## Dataset and model
- Train: 2020–2024, 1,258 chart images (1,256 labeled for binary direction).
- Validation: 2025, 250 chart images (249 labeled for binary direction).
- Final locked test: 2026, 224 images. **Not evaluated.**
- Architecture: pretrained ResNet18 with frozen image backbone and trained prediction heads; image pixels only. The chart input is resized to 336×768.
- Epochs: 8 of maximum 15; early stopping. Best checkpoint: epoch 4.
- CPU training: roughly 21 minutes, approximately 151–159 seconds per epoch.
- Run: https://github.com/jamelof23/SPY/actions/runs/45091374048

## Direction accuracy and calibration

| Metric | Actual saved 2025 validation |
|---|---:|
| Raw direction accuracy | 53.012% (132 correct of 249) |
| Temperature-calibrated direction accuracy | 53.012% |
| Validation observed green fraction | 52.610% (131/249) |
| Majority-class-only baseline accuracy | 52.610% |
| Accuracy advantage over majority baseline | **0.402 percentage points (1 extra correct prediction)** |
| Raw direction log loss | 0.689752 |
| Calibrated direction log loss | 0.689530 |
| Raw Brier score | 0.248308 |
| Calibrated Brier score | 0.248200 |
| Reference balanced 50/50 log loss | 0.693147 |

**Interpretation:** The observed validation accuracy improvement over the simple majority-class benchmark is negligible and should not be treated as evidence of a useful trading edge. Because checkpoint, threshold, and calibration were all selected on 2025, these results are *development* measurements, not an unbiased out-of-sample test.

## Predicted candle body quantiles

| Metric | Saved 2025 validation |
|---|---:|
| Number of examples | 250 |
| Average pinball loss | 0.160296 |
| Median absolute error, in past-only ATR units | 0.473139 |
| Mean nominal 10th–90th interval width, ATR | 1.040668 |
| Actual coverage of nominal 80% interval | **60.4%** |

**Interpretation:** The nominal 80% candle-body prediction range captured only 60.4% of observed validation outcomes. This is substantially undercovered; do not treat the range as a reliable 80% interval.

## Illustrative signal proxy (not executable options strategy)

| Metric | Saved 2025 validation |
|---|---:|
| CALL_BIAS signals | 74 |
| PUT_BIAS signals | **0** |
| Total directional signals | 74 |
| Share of all 250 sessions with a signal | 29.6% |
| Positive signed-SPY proxy returns among signals | 55.405% |
| Mean signed-SPY open-to-close proxy return per signal | +0.1830% |
| Mean signed-SPY proxy return per calendar session | +0.0542% |
| Assumed round-trip cost | **0 basis points** |

This *illustrative proxy* assumes buying or selling underlying SPY at the next official open and exiting at its close. It does not include realistic fills, spreads, commissions, financing, margin, options premium, Greeks, EMA-entry triggers, compounding, or real trading execution. The 74-call/0-put split suggests a strong upward bias and prevents any assessment of the model's PUT signal quality. This proxy was analyzed on the same 2025 validation period used to choose thresholds and therefore should not be taken as evidence of live profitability.

## Last historical chart
For the chart ending 2026-11-20, the model output **WAIT** with P(next green candle) approximately **0.505**, indicating no strong directional preference. This is a single unlabeled prospective output, not an accuracy test.

## Recommendation
Do not trade CALL or PUT options based on v1 results. Preserve the trained model and its configuration. Before testing 2026, specify the decision rules and costs; conduct the 2026 locked evaluation once, report direction by class, majority baseline, calibration, interval coverage, and representative cost-sensitive strategies. Do not tune the model or thresholds after seeing 2026 outcomes if claiming 2026 is an independent final test.

The machine-trained model weights are available at https://github.com/jamelof23/SPY/releases/tag/spy42-model-v1 .
