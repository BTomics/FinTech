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
    # TODO: predicted r_{t+1} = the freshest return known at close of day t.
    #   NOTE: that's today's return r_t. With the current lag convention
    #   (ret_lag1 == r_{t-1}) r_t isn't a column — see the lag-convention
    #   decision in the handoff before implementing.
    ...


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
    # TODO: per ticker (grouped — never pool tickers), expanding mean of the
    #   returns known through day t. The leakage trap: an expanding mean that
    #   sweeps in r_{t+1} (the target) is look-ahead. Make sure each row's
    #   prediction is built only from returns at or before t.
    ...
