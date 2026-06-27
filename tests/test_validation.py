import pandas as pd
import pytest
from pathlib import Path
from fintech.data.bars import parse_bars
from fintech.features.build import build_features
from fintech.models.baselines import predict_historical_mean, predict_persistence
from fintech.models.validation import (
    walk_forward_splits,
    score_predictions,
    run_walk_forward,
    run_walk_forward_model,
)
from fintech.models.gbm import fit_predict_gbm

FIXTURES = Path(__file__).parent / "fixtures"
BARS_PATH = FIXTURES / "equity_bars_sample.parquet"

def test_walk_forward_splits_causal():
    bars = parse_bars(pd.read_parquet(BARS_PATH))
    features = build_features(bars)
    splits = walk_forward_splits(features, window_length=30, horizon=1, test_size=20)
    for train_idx, test_idx in splits:
        assert max(features.loc[train_idx, "date"]) < min(features.loc[test_idx, "date"])
    
def test_walk_forward_split_gaps():
    horizon = 5
    bars = parse_bars(pd.read_parquet(BARS_PATH))
    features = build_features(bars)
    splits = walk_forward_splits(features, window_length=30, horizon=horizon, test_size=20)
    all_dates = features["date"].drop_duplicates().sort_values()
    for train_idx, test_idx in splits:
        train_idx_max = features.loc[train_idx, "date"].max()
        test_idx_min = features.loc[test_idx, "date"].min()
        gap = all_dates[(all_dates > train_idx_max) & (all_dates < test_idx_min)]
        assert len(gap) >= horizon

def test_walk_forward_split_rolling():
    window_len = 30
    bars = parse_bars(pd.read_parquet(BARS_PATH))
    features = build_features(bars)
    splits = walk_forward_splits(features, window_length=window_len, horizon=1, test_size=20)
    for train_idx, test_idx in splits:
        assert features.loc[train_idx, "date"].nunique() == window_len 

def test_walk_forward_split_no_overlap():
    bars = parse_bars(pd.read_parquet(BARS_PATH))
    features = build_features(bars)
    splits = walk_forward_splits(features, window_length=30, horizon=1, test_size=20)
    for train_idx, test_idx in splits:
        assert train_idx.intersection(test_idx).empty


def test_score_predictions_mse():
    # Hand-computed: errors 0, 2, 0 -> squared 0, 4, 0 -> mean 4/3.
    pred = pd.Series([1.0, 2.0, 3.0])
    target = pd.Series([1.0, 0.0, 3.0])
    assert score_predictions(pred, target) == pytest.approx(4 / 3)


def test_run_walk_forward_structure():
    bars = parse_bars(pd.read_parquet(BARS_PATH))
    features = build_features(bars)
    kwargs = dict(window_length=30, horizon=1, test_size=20)

    per_fold, mean = run_walk_forward(features, predict_persistence, **kwargs)
    n_splits = len(list(walk_forward_splits(features, **kwargs)))

    assert n_splits > 0                      # not vacuous
    assert len(per_fold) == n_splits         # one score per fold
    assert all(s >= 0 for s in per_fold)     # MSE is non-negative
    assert mean == pytest.approx(sum(per_fold) / len(per_fold))
    assert pd.notna(mean) and mean > 0


def test_run_walk_forward_detects_misalignment():
    bars = parse_bars(pd.read_parquet(BARS_PATH))
    features = build_features(bars)
    kwargs = dict(window_length=30, horizon=1, test_size=20)

    # Oracle: predict the realized target itself -> perfectly aligned, MSE ~ 0.
    def oracle(f):
        return f["target"]

    # Same values knocked one day out of step per ticker -> every prediction is
    # paired with the WRONG day's outcome.
    def misaligned(f):
        return f.groupby("ticker")["target"].shift(1)

    _, clean = run_walk_forward(features, oracle, **kwargs)
    _, broken = run_walk_forward(features, misaligned, **kwargs)

    assert clean == pytest.approx(0.0, abs=1e-12)
    assert broken > clean


def _gbm(train, test):
    # thin wrapper: silence LightGBM's chatter on tiny folds
    return fit_predict_gbm(train, test, verbosity=-1)


def test_run_walk_forward_model_structure():
    bars = parse_bars(pd.read_parquet(BARS_PATH))
    features = build_features(bars)
    kwargs = dict(window_length=30, horizon=1, test_size=20)

    per_fold, mean = run_walk_forward_model(features, _gbm, **kwargs)
    n_splits = len(list(walk_forward_splits(features, **kwargs)))

    assert n_splits > 0                      # not vacuous
    assert len(per_fold) == n_splits         # one score per fold
    assert all(s >= 0 for s in per_fold)     # MSE is non-negative
    assert mean == pytest.approx(sum(per_fold) / len(per_fold))
    assert pd.notna(mean) and mean > 0


def test_gbm_exploits_a_leaked_feature():
    # Leakage guard: add a feature column that IS the answer (== target). A model
    # fit through the harness should then score near-perfectly. If it does, the
    # harness/model would expose any real future leak; the fact that the CLEAN
    # features don't produce this near-zero score is the evidence they don't leak.
    bars = parse_bars(pd.read_parquet(BARS_PATH))
    features = build_features(bars)
    kwargs = dict(window_length=30, horizon=1, test_size=20)

    _, clean = run_walk_forward_model(features, _gbm, **kwargs)

    leaked = features.copy()
    leaked["leak"] = leaked["target"]  # feature_columns() will pick this up
    _, leaked_mse = run_walk_forward_model(leaked, _gbm, **kwargs)

    assert leaked_mse < clean  # the leak makes the model dramatically better
