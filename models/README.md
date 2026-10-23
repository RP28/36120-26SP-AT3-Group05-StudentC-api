Place a trusted `model.pkl` here once experiments are complete. The dev placeholder is created in code; no model.pkl is shipped.

Expected pickle payload: dict with keys `model`, `feature_columns`, `class_labels`, `forecast_horizon` (=7), optionally `preprocessor`, `version`. `model` must provide `predict()` and `predict_proba()` and use the same feature ordering as at training time. The fitted `preprocessor`, if present, must accept a one-row numeric DataFrame with exactly `feature_columns`.

If using custom transformers/functions, ensure they're importable in this project and save compatible dependency versions. Never unpickle untrusted files. `APP_PROFILE=prod` forbids a placeholder model.
