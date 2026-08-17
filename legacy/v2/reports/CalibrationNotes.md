### Calibration Notes

The Sept–Nov snapshot produces perfectly separated classes after threshold tuning (0.03 for logistic, 0.06 for Random Forest). This is expected because the snapshot mixes highly distinctive signals: GPT telemetry denial-of-wallet bursts versus clean baseline traffic, and public phishing feeds that carry overt prompt-injection strings. We document the thresholds so future freezes can be retuned if distributions shift. We also retain the default 0.5 metrics (0.983 accuracy, 6 FN) to show the model’s natural error profile.

If we capture another snapshot in the future, we will drop synthetic augmentations and include noisier telemetry to stress-test the thresholds; for now the scope stays within the Sept–Nov 2025 window.
