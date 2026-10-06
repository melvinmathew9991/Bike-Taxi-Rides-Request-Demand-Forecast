"""
Settings shared by the experiment scripts in this directory.

The sweep, the peak-error walk-forward and the weather experiment must fit the
same model on the same features, or their results cannot be compared. These
settings used to be defined in `compare_strategies.py` and imported from there
by the other two, with the XGBoost parameters written out by hand - a second
copy of `PipelineConfig.xgb_params` that nothing kept in step with the first.

Now every value is read from `PipelineConfig`, so the experiments measure what
the pipeline trains. The one deliberate difference is the tree count: the
experiments fit hundreds of models without early stopping, so they use a fixed
`SWEEP_N_ESTIMATORS` instead of the pipeline's ceiling of 600.
"""

from __future__ import annotations

from ML_Pipeline.config import PipelineConfig
from ML_Pipeline.features import build_feature_names

_CONFIG = PipelineConfig()

#: Lags and rolling window as the pipeline trains them. They were once
#: hardcoded to `(1, 2, 3)`, which is why the sweep's published numbers went
#: stale when the configured set gained the daily and weekly lags.
LAGS: tuple[int, ...] = tuple(_CONFIG.lag_features)
ROLLING_WINDOW: int = _CONFIG.rolling_window
CLUSTER_FEATURES = ("cluster_lat", "cluster_lng")


def feature_names(lags: tuple[int, ...] = LAGS) -> list[str]:
    """Feature list for the lag model, as the pipeline builds it."""
    return build_feature_names(use_lags=True, lags=lags, cluster_features=CLUSTER_FEATURES)


FEATURES: list[str] = feature_names()

#: Fixed tree count for experiments, which fit without an early-stopping tail.
SWEEP_N_ESTIMATORS = 250

#: `PipelineConfig.xgb_params` minus the objective, which each strategy sets
#: itself (`count:poisson` for a level target, `reg:squarederror` for a ratio).
BASE_PARAMS: dict = {
    **{k: v for k, v in _CONFIG.xgb_params.items() if k != "objective"},
    "n_estimators": SWEEP_N_ESTIMATORS,
}
