"""
`biketaxi stage-demo`: stage the files a hosted demo serves from - aggregated data only.

The output directory mixes what serving needs with booking-level personal data
(`clean_data_*`: rider identifiers and exact coordinates). Uploading it whole
would publish that to a bucket. This copies an allow-list instead:

    Data_Prepared_<ver>.csv.gz        demand counts per cluster per interval,
                                      trimmed to the last --history-days
    prediction_model_with_lag_<ver>   the promoted model
    pickup_cluster_model_<ver>        the 50 cluster centres ONLY - the fitted
                                      clusterer also holds a label for every
                                      training booking (3.7 M of them), which
                                      serving never reads
    model_registry.json               the promoted entry only

and refuses to stage a grid with any column outside the aggregated set.

Usage:
    biketaxi stage-demo --out deploy/.staging
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import dump, load

from ML_Pipeline.features import TS_COL
from ML_Pipeline.registry import ModelRegistry

#: Every column the demand grid may carry. Anything else stops the staging.
GRID_COLUMNS = frozenset(
    {"ts", "pickup_cluster", "request_count", "mins", "hour", "month", "quarter",
     "dayofweek"}
)
#: Serving needs max(lag) of contiguous history - 7 days - plus the rolling
#: window; two weeks leaves a margin and the forecast still starts where the
#: real history ends.
DEFAULT_HISTORY_DAYS = 14


def _find(output_dir: Path, raw: str) -> Path:
    """A registry path, or the same file name in the output directory."""
    as_written = Path(raw)
    if as_written.exists():
        return as_written
    alongside = output_dir / raw.replace("\\", "/").rsplit("/", 1)[-1]
    if alongside.exists():
        return alongside
    raise FileNotFoundError(f"Registry points at {raw}, which does not exist.")


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Stage the files a hosted demo serves from - aggregated data only.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--out", type=Path, default=Path("deploy/.staging"))
    parser.add_argument("--history-days", type=int, default=DEFAULT_HISTORY_DAYS)
    args = parser.parse_args(argv)

    registry = ModelRegistry(str(args.output_dir / "model_registry.json"))
    promoted = registry.production_model()
    if promoted is None:
        raise SystemExit("No model is promoted; promote one before staging.")
    name, info = promoted
    version = info.get("metadata", {}).get("version")
    if not version:
        raise SystemExit(f"{name} has no version in its registry metadata.")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    # Grid: allow-listed columns, last N days.
    grid_path = args.output_dir / f"Data_Prepared_{version}.csv.gz"
    grid = pd.read_csv(grid_path)
    unexpected = sorted(set(grid.columns) - GRID_COLUMNS)
    if unexpected:
        raise SystemExit(
            f"{grid_path.name} has column(s) outside the aggregated set: "
            f"{unexpected}. Not staging it."
        )
    grid[TS_COL] = pd.to_datetime(grid[TS_COL])
    cutoff = grid[TS_COL].max() - pd.Timedelta(days=args.history_days)
    recent = grid[grid[TS_COL] > cutoff]
    recent.to_csv(args.out / grid_path.name, index=False, compression="gzip")

    # Model, as trained.
    model_path = _find(args.output_dir, info["model_path"])
    shutil.copy2(model_path, args.out / model_path.name)

    # Cluster centres, without the per-booking labels.
    cluster_path = args.output_dir / f"pickup_cluster_model_{version}.joblib"
    centres = np.asarray(load(cluster_path).cluster_centers_)
    dump(SimpleNamespace(cluster_centers_=centres), args.out / cluster_path.name)

    # Registry: the promoted entry alone, pointing at the staged file.
    entry = dict(info)
    entry["model_path"] = model_path.name
    entry.setdefault("metadata", {})["version"] = version
    entry["metadata"].pop("cluster_model_path", None)
    (args.out / "model_registry.json").write_text(json.dumps({name: entry}, indent=2))

    print(f"Staged {name} to {args.out}:")
    for path in sorted(args.out.iterdir()):
        print(f"  {path.name:55} {path.stat().st_size / 1e6:6.2f} MB")
    print(
        f"Grid: {len(recent):,} rows, {recent[TS_COL].min()} to {recent[TS_COL].max()}, "
        f"{recent['pickup_cluster'].nunique()} clusters, columns {list(recent.columns)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
