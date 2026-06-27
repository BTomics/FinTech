

def walk_forward_splits(features, window_length=30, horizon=1, test_size=1):
    """Generate purged, rolling walk-forward train/test splits.

    Walks the UNIQUE DATES of the panel forward in time. Each fold trains on a
    rolling block of `window_length` consecutive dates, leaves a `horizon`-date
    purge gap, then tests on the next `test_size` dates. Splitting on dates (not
    rows) keeps every ticker's rows for a given date in the same fold, so a
    boundary never cuts one ticker's history mid-stream.

    The purge gap exists because build_features' target is r_{t+1}: the last
    training date's label reaches `horizon` days ahead, so those dates are held
    out of test to stop test-period information leaking into training.

    CONTRACT (pinned by tests/test_validation.py):
      - Causal: features.loc[train_idx, "date"].max() < .loc[test_idx,...].min()
        for every fold, and the gap between them is at least `horizon` dates.
      - Rolling: each train block spans exactly `window_length` dates.
      - Forward & disjoint: successive test blocks advance in time and never
        overlap; train_idx and test_idx share no rows.
      - Returned labels index `features` directly (features.loc[train_idx]).

    Args:
        features (pd.DataFrame): build_features output — plain int index, with
            `date` and `ticker` as COLUMNS (not a MultiIndex), sorted by
            (ticker, date).
        window_length (int): training window size, in number of unique dates.
        horizon (int): purge gap in dates between train and test; matches the
            label's look-ahead (1 for an r_{t+1} target).
        test_size (int): number of dates in each test block.

    Yields:
        tuple[pd.Index, pd.Index]: (train_idx, test_idx) row labels per fold.
    """
    # TODO: implement until tests/test_validation.py is green.
    #   1. dates = features["date"].drop_duplicates().sort_values().to_numpy()
    #   2. slide a window over `dates`: for each start, take window_length train
    #      dates, skip `horizon`, take the next `test_size` test dates; stop when
    #      the test block runs past the end.
    #   3. map date sets -> row labels:
    #        train_idx = features.index[features["date"].isin(train_dates)]
    #        test_idx  = features.index[features["date"].isin(test_dates)]
    #   4. yield (train_idx, test_idx).
    dates = features["date"].drop_duplicates().sort_values().to_numpy()
    for i in range(len(dates)):
        train_dates = dates[i:i+window_length]
        test_dates = dates[i+window_length+horizon:i+window_length+horizon+test_size]
        if len(test_dates) == 0:
            break
        train_idx = features.index[features["date"].isin(train_dates)]
        test_idx  = features.index[features["date"].isin(test_dates)]
        yield (train_idx, test_idx)

def score_predictions(pred, target):
    """
    Mean squared error of one prediction slice against its target.

    Dumb on purpose: no slicing, no folds — just the metric over two aligned
    Series. The caller (run_walk_forward) decides what slice to pass. MSE first;
    directional accuracy / IC can be added later once we know what to optimize.

    Args:
        pred (pd.Series): predicted next-day returns.
        target (pd.Series): realized next-day returns, aligned to `pred`.

    Returns:
        float: mean squared error.
    """
    mse = ((pred - target) ** 2).mean()
    return mse


def run_walk_forward(features, predict_fn, **split_kwargs):
    """
    Score a prediction function across purged walk-forward folds.

    For each fold from walk_forward_splits, score predict_fn's predictions on the
    test rows against their realized target. Returns the per-fold scores AND the
    mean, so a baseline and a model run through the exact same splits are
    directly comparable (M3 success criterion).

    NOTE — current seam: predict_fn(features) -> Series aligned to features.index.
    This works because the baselines are causal BY CONSTRUCTION (a prediction at
    row t never reads a future row), so computing once over the whole panel and
    slicing the test fold == computing online. LightGBM is NOT causal by
    construction and will need a fit-on-train-only variant; we extend the seam to
    predict_fn(train, test) when it lands.

    Args:
        features (pd.DataFrame): build_features output (has `target`).
        predict_fn (callable): features -> pd.Series of predictions.
        **split_kwargs: forwarded to walk_forward_splits (window_length, horizon,
            test_size).

    Returns:
        tuple[list[float], float]: (per-fold MSEs, mean MSE).
    """
    
    preds = predict_fn(features)
    per_fold = []
    for train_idx, test_idx in walk_forward_splits(features, **split_kwargs):
        score = score_predictions(preds.loc[test_idx],
                                  features.loc[test_idx, "target"])
        per_fold.append(score)
    return per_fold, sum(per_fold) / len(per_fold)