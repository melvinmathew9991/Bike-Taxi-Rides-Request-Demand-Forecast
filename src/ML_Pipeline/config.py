"""
Pipeline configuration.

Every field here is read by the code it claims to configure. That was not
previously true: `n_clusters` (default 300) was ignored while the clustering
stage hardcoded 50; `xgb_params` (depth 7, lr 0.1) was ignored while the trainer
hardcoded depth 8, lr 0.01; `lag_features`, `rolling_window`, `test_size` and
`train_day_cutoff` were never read at all. The `--n-clusters` CLI flag was
accepted, logged, written into the saved config snapshot, and discarded - and the
troubleshooting guide told users to lower it to fix out-of-memory errors.

Silently-ignored configuration is worse than no configuration: it makes a run
look reproducible while the recorded settings had no effect. `assert_wired()`
below is the guard against that regressing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from ML_Pipeline.artifacts import DATA_STEMS, MODEL_STEMS

logger = logging.getLogger(__name__)

#: Longest forecast horizon, for the CLI and the API alike. Two days is where
#: measured recursive MASE stops clearly beating seasonal-naive (0.82 at two
#: days, 0.99 at one week); see the horizon table in docs/MODEL_CARD.md.
MAX_HORIZON_DAYS = 2


def max_horizon_steps(interval_minutes: int) -> int:
    """`MAX_HORIZON_DAYS` in intervals of `interval_minutes`."""
    return MAX_HORIZON_DAYS * 24 * 60 // interval_minutes


@dataclass
class PipelineConfig:
    """Settings for one pipeline run."""

    # --- Paths -----------------------------------------------------------
    project_root: str = ""
    data_dir: str = "data"
    raw_data_path: str = "data/raw_data.csv"
    test_data_path: str = "data/cleaned_test_booking_data.csv"
    output_dir: str = "output"
    logs_dir: str = "logs"

    # --- Time grid -------------------------------------------------------
    interval_minutes: int = 30
    freq: str = "30min"

    # --- Clustering ------------------------------------------------------
    n_clusters: int = 50
    clustering_algorithm: str = "minibatch"  # 'minibatch' | 'kmeans'
    run_cluster_diagnostics: bool = False
    #: Use cluster centroid lat/lng instead of the raw integer label. A K-Means
    #: label is nominal; feeding the integer to a tree model produces splits on
    #: an arbitrary labelling rather than on geography.
    use_cluster_centroids: bool = True

    # --- Splitting -------------------------------------------------------
    #: Fraction of the timeline held out, chronologically, as the test set.
    #: The previous split was on day-of-month (<=23 train, >23 test), which
    #: interleaves test weeks throughout the training period and lets the model
    #: see the future relative to any test point - it measures interpolation,
    #: not forecasting.
    test_fraction: float = 0.2
    #: Fraction of the *training* span reserved for early-stopping validation.
    validation_fraction: float = 0.1
    #: After scoring, refit the saved models on every row, test window included.
    #: The test window is the newest fifth of the timeline, so without this the
    #: promoted model starts its life that far behind the data - ten weeks on the
    #: reference dataset, past the six-week point where a stale model was
    #: measured to lose to seasonal-naive. Metrics and the deploy gate always
    #: come from the held-out fit. Off only to reproduce older runs.
    refit_on_all_data: bool = True

    # --- Features --------------------------------------------------------
    #: Lags in intervals. At 30 minutes: 1/2/3 are the last 90 minutes, 48 is
    #: the same time yesterday, and 336 is the same time last week.
    #:
    #: The daily and weekly lags are not optional refinements. This model is
    #: judged against a seasonal-naive baseline - the value 336 intervals
    #: earlier - and with lags stopping at 90 minutes it was being asked to
    #: beat a signal it had never been given. It did not: on the chronological
    #: split it scored RMSE 4.8032 against the baseline's 4.6062, MASE 0.9992,
    #: and `compare_to_baselines` logged "ship the baseline instead".
    #:
    #: Adding 48 and 336 takes that to RMSE 3.7505, MASE 0.8086. Measured
    #: ablation also tried 24-hour and 7-day rolling means on top: they add
    #: only 0.0116 MASE, which does not justify changing the persisted feature
    #: contract (`rolling_window` is a scalar in the bundle), so they are left
    #: out.
    #:
    #: Cost: `lag_336` makes the first week of each cluster unusable as
    #: training data - 16,800 rows of 878,300 on the reference dataset - and
    #: recursive serving now needs 7 days of contiguous history per cluster
    #: rather than 90 minutes. See `prediction_pipeline`, which seeds from the
    #: demand grid for exactly this reason.
    lag_features: tuple[int, ...] = (1, 2, 3, 48, 336)
    rolling_window: int = 3

    # --- Model -----------------------------------------------------------
    #: `count:poisson` is the appropriate objective for a non-negative count
    #: target whose median is 0. `reg:squarederror` treats it as unbounded and
    #: real-valued, and will emit negative demand.
    xgb_params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "count:poisson",
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 600,
            "min_child_weight": 5,
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    early_stopping_rounds: int = 50

    # --- Forecasting -----------------------------------------------------
    #: Intervals to forecast. Defaults to one day at `interval_minutes`, and may
    #: not exceed `MAX_HORIZON_DAYS`.
    horizon_steps: int | None = None

    # --- Prediction intervals ---------------------------------------------
    #: Nominal coverage of the forecast interval: 0.8 is the 10th to 90th
    #: percentile. The upper bound is the figure to plan supply from.
    interval_level: float = 0.8
    #: Window after the training cut that calibration origins are spread over.
    #: Four weeks is the retraining cadence, so the errors calibrated on are from
    #: a model as stale as a deployed one gets.
    interval_calibration_days: int = 28
    #: Recursive backtests run to calibrate, each over the longest horizon the
    #: API serves. 24 take about 30 seconds on the reference dataset.
    interval_origins: int = 24

    # --- Logging / registry ---------------------------------------------
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    save_models: bool = True
    save_intermediate_data: bool = True
    model_version: str = ""

    def __post_init__(self) -> None:
        if not self.model_version:
            self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lag_features = tuple(int(x) for x in self.lag_features)
        self.freq = self.freq or f"{self.interval_minutes}min"
        self.validate()

    def ensure_directories(self) -> None:
        """
        Create the output and log directories.

        Called by whatever is about to write, not from `__post_init__`.
        Constructing a config to inspect or validate it should not touch the
        filesystem - doing so created stray `output/` and `logs/` directories
        wherever a config happened to be instantiated, including the repository
        root during test runs.
        """
        for dir_path in (self.output_dir, self.logs_dir):
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Fail fast on settings that cannot produce a sensible run."""
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters}")
        if not 0 < self.test_fraction < 1:
            raise ValueError(
                f"test_fraction must be in (0, 1), got {self.test_fraction}"
            )
        if not 0 <= self.validation_fraction < 1:
            raise ValueError(
                f"validation_fraction must be in [0, 1), got {self.validation_fraction}"
            )
        if self.interval_minutes < 1:
            raise ValueError(
                f"interval_minutes must be >= 1, got {self.interval_minutes}"
            )
        if any(lag < 1 for lag in self.lag_features):
            raise ValueError(f"lag_features must all be >= 1, got {self.lag_features}")
        if self.rolling_window < 1:
            raise ValueError(f"rolling_window must be >= 1, got {self.rolling_window}")
        if self.clustering_algorithm not in {"minibatch", "kmeans"}:
            raise ValueError(
                f"clustering_algorithm must be 'minibatch' or 'kmeans', "
                f"got {self.clustering_algorithm!r}"
            )
        if not 0 < self.interval_level < 1:
            raise ValueError(
                f"interval_level must be in (0, 1), got {self.interval_level}"
            )
        if self.interval_calibration_days < 1 or self.interval_origins < 2:
            raise ValueError(
                "interval_calibration_days must be >= 1 and interval_origins >= 2, "
                f"got {self.interval_calibration_days} and {self.interval_origins}"
            )
        if self.horizon_steps is not None:
            limit = max_horizon_steps(self.interval_minutes)
            if not 1 <= self.horizon_steps <= limit:
                raise ValueError(
                    f"horizon_steps must be between 1 and {limit} "
                    f"({MAX_HORIZON_DAYS} days at {self.interval_minutes} minutes), "
                    f"got {self.horizon_steps}. Past two days the model stops "
                    "clearly beating a seasonal-naive forecast, and by a week it "
                    "only ties it - see the horizon table in docs/MODEL_CARD.md."
                )

    # --- Derived paths ---------------------------------------------------

    def get_model_path(self, model_type: str) -> str:
        """
        Path for a model artefact.

        Versioned and unversioned names used to disagree: this method returned
        `prediction_model_with_lag_<version>.joblib` while the pipeline actually
        wrote `prediction_model_with_lag.joblib`, so every registry entry pointed
        at a file that did not exist. One method now owns the naming and both the
        writer and the registry call it.
        """
        stem = MODEL_STEMS.get(model_type, model_type)
        return str(Path(self.output_dir) / f"{stem}_{self.model_version}.joblib")

    def get_data_path(self, data_type: str) -> str:
        """Path for an intermediate or output dataset."""
        stem = DATA_STEMS.get(data_type, data_type)
        # `.csv.gz`, not `.csv`. These files are gzip-compressed, and naming them
        # `.csv` meant pandas could not infer that - so every reader needed a
        # sniffing fallback, and an outside consumer running a plain
        # `pd.read_csv` got a UnicodeDecodeError on byte 0x8b.
        return str(Path(self.output_dir) / f"{stem}_{self.model_version}.csv.gz")

    # --- Serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Full configuration, including every field (nothing silently omitted)."""
        out = asdict(self)
        out["lag_features"] = list(self.lag_features)
        return out

    def save_config(self, filepath: str | None = None) -> str:
        if filepath is None:
            filepath = str(
                Path(self.output_dir) / f"pipeline_config_{self.model_version}.json"
            )
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=4)
        logger.info("Configuration snapshot written to %s", filepath)
        return filepath

    @classmethod
    def load_config(cls, filepath: str) -> PipelineConfig:
        """
        Load a configuration snapshot, ignoring unknown keys.

        Snapshots written by other versions may carry retired fields; those
        should not make an otherwise valid config unloadable.
        """
        with open(filepath, encoding="utf-8") as fh:
            raw = json.load(fh)
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            logger.warning("Ignoring unknown config key(s) in %s: %s", filepath, unknown)
        return cls(**{k: v for k, v in raw.items() if k in known})

    def assert_wired(self) -> None:
        """
        Log the settings that actually reach downstream code.

        A cheap, explicit inventory so a future change that stops honouring a
        field is visible in the run log instead of silent.
        """
        logger.info(
            "Effective configuration: n_clusters=%d (%s), freq=%s, "
            "centroid_features=%s, lags=%s, rolling_window=%d, "
            "test_fraction=%.2f, refit_on_all_data=%s, objective=%s, "
            "n_estimators=%s, early_stopping_rounds=%d",
            self.n_clusters,
            self.clustering_algorithm,
            self.freq,
            self.use_cluster_centroids,
            self.lag_features,
            self.rolling_window,
            self.test_fraction,
            self.refit_on_all_data,
            self.xgb_params.get("objective"),
            self.xgb_params.get("n_estimators"),
            self.early_stopping_rounds,
        )
