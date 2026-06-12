"""Feed-agnostic raw snapshot storage (guidelines §5: raw data is immutable)."""

import json
from pathlib import Path


def save_snapshot(payload, snapshot_time, output_dir):
    """
    Save a JSON-serializable payload to a time-stamped json file in the given directory.
    Within one output directory, one timestamp = one snapshot, and a collision is an error.
    Args:
        payload (list | dict): raw API data, saved verbatim.
        snapshot_time (pd.Timestamp): snapshot timestamp.
        output_dir (pathlib.Path or str): where to save the file.

    Returns:
        pathlib.Path: absolute path to the created json file.
    Raises:
        FileExistsError: if the same file already exists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    filename = snapshot_time.strftime("%Y%m%dT%H%M%SZ.json")
    filepath = output_dir / filename
    if filepath.exists():
        raise FileExistsError(f"File already exists: {filepath}")

    with open(filepath, "x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return filepath
