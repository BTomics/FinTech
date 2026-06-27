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
import numpy as np
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


def walk_forward_predict(features, retrain_every=252, min_train=504, **params):
    """
    Out-of-sample predictions over the whole panel via expanding-window refits.

    Walks the unique dates forward: every `retrain_every` dates, refit on ALL
    rows strictly BEFORE the current block, then predict that block. Causal —
    each prediction uses only earlier dates. Refitting periodically (not per day)
    keeps it tractable on a wide panel while staying leak-free.

    Args:
        features (pd.DataFrame): build_features output (date, ticker, feats, target).
        retrain_every (int): refit cadence, in unique dates.
        min_train (int): dates of history required before the first prediction.
        **params: passed to the LightGBM regressor.

    Returns:
        pd.Series: predictions aligned to features.index; NaN before min_train.
    """
    cols = feature_columns(features)
    dates = np.sort(features["date"].unique())
    preds = pd.Series(np.nan, index=features.index)
    for start in range(min_train, len(dates), retrain_every):
        split_date = dates[start]
        block = dates[start:start + retrain_every]
        train = features[features["date"] < split_date]   # strictly earlier
        test = features[features["date"].isin(block)]
        model = lgb.LGBMRegressor(**params, verbose=-1).fit(train[cols], train["target"])
        preds.loc[test.index] = model.predict(test[cols])
    return preds
