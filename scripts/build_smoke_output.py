#!/usr/bin/env python
"""
Build a small synthetic output directory the serving container can load.

For the container smoke test in CI: a demand grid, a real (tiny) XGBoost model
in a `ModelBundle`, and a registry with it promoted. Nothing is read from disk,
so it runs on a bare checkout, and no real data - personal or aggregated - goes
anywhere near the image.

CI runs it inside the built image, so the model is pickled by the same xgboost
the container will unpickle it with:

    docker run --rm -i --user "$(id -u)" -v "$PWD/smoke:/smoke" IMAGE \\
        python - /smoke < scripts/build_smoke_output.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump

from ML_Pipeline.features import (
    CLUSTER_COL,
    TARGET_COL,
    TS_COL,
    ModelBundle,
    add_lag_features,
)
from ML_Pipeline.registry import ModelRegistry

VERSION = "20210101_000000"
CLUSTERS = 3
INTERVALS = 336 * 3  # three weeks: enough for the weekly lag and a fit
FEATURES = ["lag_1", "lag_336", "rolling_mean"]


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    stamps = pd.date_range("2021-01-01", periods=INTERVALS, freq="30min")
    grid = pd.DataFrame(
        [
            {TS_COL: t, CLUSTER_COL: c,
             TARGET_COL: float(rng.poisson(2 + c + 3 * (8 <= t.hour <= 20)))}
            for c in range(CLUSTERS)
            for t in stamps
        ]
    )
    grid.to_csv(out / f"Data_Prepared_{VERSION}.csv.gz", index=False,
                compression="gzip")

    lagged = add_lag_features(grid, lags=(1, 336), rolling_window=3)
    model = xgb.XGBRegressor(objective="count:poisson", n_estimators=20, max_depth=3)
    model.fit(lagged[FEATURES], lagged[TARGET_COL])
    metrics = {"test_rmse": 1.0, "baseline_mase": 0.9, "beats_seasonal_naive": 1.0,
               "clusters_losing_to_naive": 0.0}
    bundle = ModelBundle(
        model=model, feature_names=FEATURES, uses_lags=True, lags=(1, 336),
        rolling_window=3, freq="30min", trained_at=datetime.now().isoformat(),
        data_through=str(stamps[-1]), metrics=metrics,
        notes="Synthetic smoke-test model. Not a forecast of anything.",
    )
    # Recorded the way a Windows training run records it, so the smoke test
    # also proves a registry written on another OS loads in the container.
    model_file = f"prediction_model_with_lag_{VERSION}.joblib"
    dump(bundle, out / model_file)
    registry = ModelRegistry(str(out / "model_registry.json"))
    name = f"xgb_with_lag_{VERSION}"
    registry.register_model(
        model_name=name, model_path=f"output\\{model_file}",
        model_type="xgboost", metrics=metrics, metadata={"version": VERSION},
    )
    registry.promote_model(name)
    print(f"Smoke output written to {out}: {len(grid):,} grid rows, model {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "smoke_output")))
