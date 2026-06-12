"""Tests for fintech.data.storage.

save_snapshot(payload, snapshot_time, output_dir) writes the payload verbatim
to <output_dir>/<compact-timestamp>.json; a timestamp collision is an error.
"""

import json

import pandas as pd
import pytest

from fintech.data.storage import save_snapshot

PAYLOAD = [{"id": "1", "question": "test?"}]
T18 = pd.Timestamp("2026-06-12T18:00:00Z")
T19 = pd.Timestamp("2026-06-12T19:00:00Z")


def test_onefile_outputdir(tmp_path):
    save_snapshot(PAYLOAD, T18, tmp_path)
    assert (tmp_path / "20260612T180000Z.json").exists()
    assert len(list(tmp_path.iterdir())) == 1

def test_twofiles_difftimes(tmp_path):
    save_snapshot(PAYLOAD, T18, tmp_path)
    save_snapshot(PAYLOAD, T19, tmp_path)
    assert (tmp_path / "20260612T180000Z.json").exists()
    assert (tmp_path / "20260612T190000Z.json").exists()

def test_twofiles_loudfail(tmp_path):
    save_snapshot(PAYLOAD, T18, tmp_path)
    with pytest.raises(FileExistsError):
        save_snapshot(PAYLOAD, T18, tmp_path)

def test_file_contains_input(tmp_path):
    path = save_snapshot(PAYLOAD, T18, tmp_path)
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == PAYLOAD

def test_dict_payload_roundtrip(tmp_path):
    payload = {"KXFED": [{"ticker": "KXFED-27APR"}], "KXCPI": []}
    path = save_snapshot(payload, T18, tmp_path)
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == payload
