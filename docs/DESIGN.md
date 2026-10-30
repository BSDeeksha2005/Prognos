# Prognos design

Training labels are remaining cycles to failure clipped at 125. The cap follows a common piecewise target convention: early-life cycles have little observable degradation signal, and clipping reduces their influence. Each training label is computed from that engine's own last cycle.

The validation split groups whole engines before model selection. Rows from one trajectory are correlated, so a row split would leak engine-specific patterns into both sides. Rolling statistics are trailing windows. Near-constant sensor and engineered-feature filtering uses raw training-split variance with a fixed `1e-8` threshold, before scaling; standardization is then fitted only on retained training features. Test engines are used only for terminal-cycle final scoring. The unit-level assignment test guards the most important split invariant.

Linear regression is a simple reference; LightGBM captures nonlinear sensor interactions in engineered tabular features. A small 1D convolution learns local patterns from trailing windows. Early stopping monitors validation engines. Five fixed hyperparameter configurations are defined. A seeded run is used to reduce accidental randomness, although GPU kernels may not be bit-identical.

The NASA score is asymmetric: overestimating remaining life is penalized more heavily than underestimating it. PSI summarizes bucket-share shifts, and the two-sample KS test checks distribution separation. The report compares selected raw sensor distributions in the training split against the held-out test inputs; either configured threshold flags a sensor. This is a benchmark train/test comparison, not live fleet monitoring, and its thresholds are heuristics that would need calibration against representative operating history.

ONNX provides a portable inference artifact. The export check compares PyTorch and ONNX Runtime on an identical tensor. The HTTP adapter validates a 30-cycle by 21-sensor window and loads the model once during application lifespan.

The web console separates the overview, per-engine registry, terminal-cycle RUL comparison, model evaluation, and source/drift checks. It shows values only when the API returns a completed evaluation artifact; missing files, offline service, and unrun models have distinct states. Registry ordering is an inspection aid based on estimates, not an action recommendation.

Problem framing: this prototype explores ranking engine runs for inspection planning and estimating remaining cycles from recent sensor windows. A future aerospace user might adapt this workflow to propulsion test stands or ground equipment, but only after training and validation on that system's own data.

Limitations: data are simulated, FD001 is turbofan data with one operating condition and one fault mode, and this work has no real-fleet or spacecraft-propulsion validation. It is a research prototype, not an operational or flight-certified tool. As the requested NASA files were absent, no baseline/model results or ONNX artifact could be measured in this workspace. FD002–FD004 support and production monitoring policy remain future work.
