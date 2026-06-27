import pandas as pd
from pathlib import Path
from fintech.data.bars import parse_bars
from fintech.features.build import build_features
from fintech.models.baselines import predict_historical_mean, predict_persistence

FIXTURES = Path(__file__).parent / "fixtures"
BARS_PATH = FIXTURES / "equity_bars_sample.parquet"

def test_index_alignment():
    bars = parse_bars(pd.read_parquet(BARS_PATH))
    features = build_features(bars)
    prediction= predict_persistence(features)
    assert prediction.index.equals(features.index)
    prediction= predict_historical_mean(features)
    assert prediction.index.equals(features.index)

def test_persistence():
    bars = parse_bars(pd.read_parquet(BARS_PATH))
    features = build_features(bars)
    prediction= predict_persistence(features)
    assert (prediction == features["ret_lag1"]).all()

def test_mean_leakage():
    bars = parse_bars(pd.read_parquet(BARS_PATH))  
    features = build_features(bars)
    prediction= predict_historical_mean(features)
    tampered = features.copy()
    tampered["target"] = 0
    tampered_prediction = predict_historical_mean(tampered)
    pd.testing.assert_series_equal(prediction, tampered_prediction)

def test_historical_mean_value():
    bars = parse_bars(pd.read_parquet(BARS_PATH))
    features = build_features(bars)
    prediction = predict_historical_mean(features)

    expected = (
        features.groupby("ticker")["ret_lag1"]
        .expanding().mean()
        .reset_index(level=0, drop=True)
    )
    pd.testing.assert_series_equal(prediction, expected, check_names=False)


   
    
