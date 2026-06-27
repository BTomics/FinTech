"""Naive return-prediction baselines (M3).

The yardsticks every real model has to beat. If LightGBM can't outperform
"tomorrow looks like the recent past," it isn't adding value and doesn't stay
(development_plan M3 + standing rules).

Each baseline takes the supervised frame from features.build_features and
returns predicted next-day returns aligned to its rows, so a model and a
baseline can be scored against the same `target` on identical walk-forward
splits.
"""

import pandas as pd


def predict_persistence(features):
    """
    Persistence baseline: tomorrow's return == the most recent observed return.

    The simplest possible forecast — no fitting, just carry the freshest return
    forward. Causal by construction: it uses only a value already known at the
    close of day t.

    Args:
        features (pd.DataFrame): build_features output (date, ticker, feature
            columns, target).

    Returns:
        pd.Series: predicted next-day return, indexed like `features`.
    """
    
    prediction = features["ret_lag1"]
    return prediction.reindex(features.index)

def predict_historical_mean(features):
    """
    Historical-mean baseline: tomorrow's return == mean of all returns so far.

    Predicts the expanding (causal) average return per ticker — it must never
    include the current target or any future return.

    Args:
        features (pd.DataFrame): build_features output.

    Returns:
        pd.Series: predicted next-day return, indexed like `features`.
    """
    prediction = (
        features.groupby("ticker")["ret_lag1"]
        .expanding().mean()
        .reset_index(level=0, drop=True)
    )
    return prediction
    
