"""Gradient-boosted-tree return model (M3).

The first non-trivial model in the pipeline. It only earns its place if it beats
the naive baselines (persistence, historical mean) on the SAME purged
walk-forward splits — otherwise it's removed, not kept "just in case"
(development_plan M3 + standing rules).

Unlike the baselines this is NOT causal by construction: it must be fit on a
fold's training rows only and then asked to predict that fold's test rows, which
is why it plugs into run_walk_forward_model's (train, test) seam rather than the
whole-panel run_walk_forward.
"""

import lightgbm as lgb
import pandas as pd
# Everything in build_features output that isn't an identifier or the label.
NON_FEATURES = ("date", "ticker", "target")


def feature_columns(features):
    """Return the model's input column names (all cols except id/label)."""
    return [c for c in features.columns if c not in NON_FEATURES]


def fit_predict_gbm(train, test, **params):
    """
    Fit a LightGBM regressor on `train`, predict next-day return for `test`.

    The fold-local fit/predict step driven by run_walk_forward_model. Trains only
    on `train` (no peeking at `test` or the future) and returns predictions
    aligned to `test`'s rows so they can be scored against test's `target`.

    Args:
        train (pd.DataFrame): training fold (feature cols + `target`).
        test (pd.DataFrame): test fold (same feature cols; `target` not used here).
        **params: passed through to the LightGBM regressor.

    Returns:
        pd.Series: predicted next-day returns, indexed like `test`.
    """
    # TODO: pick the feature columns and the target; fit a LightGBM regressor on
    #   the train fold; predict the test fold's features; return the predictions
    #   as a Series carrying test's index (so scoring aligns).
    
    features_cols = feature_columns(train)
    train_x = train[features_cols]
    train_y = train["target"]
    test_x = test[features_cols]

    model = lgb.LGBMRegressor(**params, verbose=-1)

    model.fit(
        train_x,
        train_y,
    )

    preds = model.predict(test_x)

    return pd.Series(preds, index=test.index)
