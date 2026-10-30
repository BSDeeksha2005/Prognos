# Learning guide

## Design choices and concepts

- **Leakage:** leakage is information from validation or test that sneaks into training. It makes evaluation look better than future performance. Prognos fits scaling and feature filtering on training data only and uses trailing windows.
- **Overfitting:** a model overfits when it memorizes its training examples instead of learning patterns that transfer. Validation engines and early stopping help reveal this.
- **Engine split:** all cycles from one engine stay in one split because cycles from the same engine share a trajectory and are not independent samples.
- **RUL clipping:** early-life cycles can have indistinguishable health signals despite large RUL values. A cap limits their effect and uses the requested piecewise target.
- **Scaling:** standardization recenters and rescales features. Fitting its mean and scale on training rows prevents validation/test information from influencing preprocessing.
- **Early stopping:** training stops after validation loss stops improving, limiting unnecessary fitting to the training trajectories.
- **PSI:** Population Stability Index compares how samples fall into reference-defined buckets. It is an interpretable distribution-shift signal, not proof that a model is wrong.
- **KS:** the two-sample Kolmogorov–Smirnov test compares empirical distributions. Its statistic is a maximum CDF gap; its p-value depends on sample assumptions and size.
- **ONNX:** ONNX is a portable format for exchanging neural models. Runtime parity checks that conversion preserved the prediction for a sample input.
- **Asymmetric NASA score:** the score weights overestimated RUL more heavily, reflecting the cost of predicting more life than an engine actually has.

## Interview questions

1. **Why split by engine?** To evaluate on trajectories the model did not see during fitting.
2. **What is target leakage?** Using information unavailable at prediction time or derived from evaluation data during training.
3. **Why cap RUL?** It limits the effect of early-life labels that lack a distinct degradation signature.
4. **Why calculate rolling statistics?** They summarize recent behavior while retaining the current cycle's local temporal context.
5. **How do you avoid future-cycle leakage?** Features for a cycle use only that cycle and earlier observations from its engine.
6. **Why fit a scaler on train only?** Validation and test distributions must not influence learned preprocessing parameters.
7. **What does early stopping do?** It retains weights from the best validation epoch and stops after validation stops improving.
8. **Why use RMSE and the NASA score?** RMSE measures average squared-error scale; the asymmetric score captures different operational costs by error direction.
9. **What do PSI and KS tell you?** They indicate distribution change using bucket proportions and empirical CDF distance, respectively.
10. **Why export ONNX?** It separates model training from serving and allows inference in a portable runtime.

## Resume bullets

No real benchmark run was possible because the FD001 files were absent, so these bullets make no performance claims:

- Built an engine-level, leakage-aware NASA C-MAPSS RUL pipeline with train-only preprocessing.
- Implemented linear and LightGBM baselines, windowed 1D-CNN training, and validation-based model selection.
- Added ONNX inference serving, input validation, and PSI/KS sensor drift monitoring.
