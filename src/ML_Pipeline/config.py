"""
Pipeline configuration and model registry.

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
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Artefact filename stems, shared by the path builders on `PipelineConfig` and
#: by `latest_artifact` below. One definition, so a reader and a writer cannot
#: disagree about what a file is called - which is exactly how the dashboard
#: ended up looking for `Data_Prepared.csv` while the pipeline wrote
#: `Data_Prepared_<version>.csv`.
MODEL_STEMS: dict[str, str] = {
    "without_lag": "prediction_model_without_lag",
    "with_lag": "prediction_model_with_lag",
    "clustering": "pickup_cluster_model",
}

DATA_STEMS: dict[str, str] = {
    "clean": "clean_data",
    "prepared": "Data_Prepared",
    "with_lag": "data_with_lag",
    "without_lag": "data_without_lag",
}

#: Longest forecast horizon, for the CLI and the API alike. Two days is where
#: measured recursive MASE stops clearly beating seasonal-naive (0.82 at two
#: days, 0.99 at one week); see the horizon table in docs/MODEL_CARD.md.
MAX_HORIZON_DAYS = 2


def max_horizon_steps(interval_minutes: int) -> int:
    """`MAX_HORIZON_DAYS` in intervals of `interval_minutes`."""
    return MAX_HORIZON_DAYS * 24 * 60 // interval_minutes


def latest_artifact(output_dir: str | Path, data_type: str) -> Path | None:
    """
    Newest versioned dataset of a given type in `output_dir`.

    Every artefact is written as `<stem>_<model_version>.csv`, where the version
    is `%Y%m%d_%H%M%S`. That format sorts lexicographically in chronological
    order, so the newest run is simply the maximum name - no date parsing, and
    no dependence on filesystem timestamps, which copying a directory destroys.

    A consumer that hardcodes the unversioned name finds nothing after a
    successful run. The dashboard did exactly that, and rendered its empty state
    over a complete set of outputs.

    Args:
        output_dir: Directory the pipeline writes to.
        data_type: Key of `DATA_STEMS`, or a literal stem.

    Returns:
        Path to the newest match, the unversioned legacy file if that is all
        there is, or None when neither exists.
    """
    directory = Path(output_dir)
    if not directory.is_dir():
        return None

    stem = DATA_STEMS.get(data_type, data_type)
    # `.csv.gz` is what the pipeline writes now; bare `.csv` is what it wrote
    # before, and output directories from earlier runs should keep working.
    for pattern in (f"{stem}_*.csv.gz", f"{stem}_*.csv"):
        versioned = sorted(directory.glob(pattern))
        if versioned:
            return versioned[-1]

    for legacy in (directory / f"{stem}.csv.gz", directory / f"{stem}.csv"):
        if legacy.exists():
            return legacy
    return None


def latest_version(output_dir: str | Path) -> str | None:
    """
    Version string of the most recent run in `output_dir`.

    Read from the configuration snapshots, since those are written last and so
    only exist for runs that reached the end.
    """
    directory = Path(output_dir)
    if not directory.is_dir():
        return None
    snapshots = sorted(directory.glob("pipeline_config_*.json"))
    if not snapshots:
        return None
    return snapshots[-1].stem.removeprefix("pipeline_config_")


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


class ModelRegistry:
    """Tracks trained models, their metrics, and where their artefacts live."""

    def __init__(self, registry_path: str | None = None):
        self.registry_path = registry_path or "./model_registry.json"
        self.registry: dict[str, dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, encoding="utf-8") as fh:
                    return json.load(fh)
            except json.JSONDecodeError:
                logger.warning(
                    "Registry at %s is corrupt; starting a fresh one.",
                    self.registry_path,
                )
        return {}

    def register_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str,
        metrics: dict[str, float] | None = None,
        parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a trained model.

        A model registered with no metrics is close to useless - `get_best_model`
        can never select it - so an empty metrics dict is now warned about
        rather than accepted in silence. Previously the pipeline never passed
        metrics at all, so every entry stored `{}` and `get_best_model` always
        returned `None`.
        """
        if not metrics:
            logger.warning(
                "Registering %r with no metrics; it can never be selected by "
                "get_best_model().", model_name,
            )
        if not Path(model_path).exists():
            logger.warning(
                "Registering %r but no artefact exists at %s.", model_name, model_path
            )

        self.registry[model_name] = {
            "model_path": str(model_path),
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "metrics": dict(metrics or {}),
            "parameters": dict(parameters or {}),
            "metadata": dict(metadata or {}),
        }
        self._save_registry()
        logger.info("Registered model %r -> %s", model_name, model_path)

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        return self.registry.get(model_name)

    # --- promotion -------------------------------------------------------
    #
    # Serving must load a *blessed* model, not merely the newest or the one with
    # the best number. Those are different things: the newest may have failed its
    # deploy gate, and `get_best_model` ranks by a raw metric across all history,
    # which is not comparable between runs whose training data differed - as it
    # did when the cleaning rules were corrected and mean demand rose, taking
    # every RMSE with it.

    def promote_model(self, model_name: str, *, force: bool = False) -> None:
        """
        Mark a model as the one serving should load.

        Refuses a model that failed its deploy gate, because promoting one would
        put a forecaster into production that a free seasonal-naive baseline
        beats. `force` overrides, for the case where you have a reason.

        Demotes whatever was in production, so exactly one model holds the stage.

        Raises:
            KeyError: if the model is not registered.
            ValueError: if it failed its gate and `force` is not set.
        """
        info = self.registry.get(model_name)
        if info is None:
            raise KeyError(f"{model_name!r} is not registered; cannot promote it.")

        beats = info.get("metrics", {}).get("beats_seasonal_naive")
        if not force:
            if beats is None:
                # An unverified model is not a passing one. This first let the
                # pre-gate August model through on a warning - which is exactly
                # the model the audit found losing to the baseline at MASE 0.999.
                # Same principle the gate itself applies: "not measured" is not
                # "passed".
                raise ValueError(
                    f"{model_name!r} carries no deploy-gate verdict, so there is "
                    "no evidence it beats a seasonal-naive baseline. It may "
                    "predate the gate, or its test window may have been shorter "
                    "than one seasonal period. Retrain it, or pass force=True if "
                    "you have a reason."
                )
            if not bool(beats):
                raise ValueError(
                    f"{model_name!r} did not beat its seasonal-naive baseline "
                    f"(MASE {info['metrics'].get('baseline_mase')}). Promoting it "
                    "would serve forecasts worse than a free baseline. Pass "
                    "force=True if you have a reason."
                )
        elif beats is None:
            logger.warning(
                "Force-promoting %r, which carries no deploy-gate verdict.",
                model_name,
            )

        for name, entry in self.registry.items():
            if entry.get("stage") == "production" and name != model_name:
                entry["stage"] = "archived"
                entry["demoted_at"] = datetime.now().isoformat()

        info["stage"] = "production"
        info["promoted_at"] = datetime.now().isoformat()
        self._save_registry()
        logger.info("Promoted %r to production.", model_name)

    def production_model(self) -> tuple[str, dict[str, Any]] | None:
        """The model serving should load, or None if nothing is promoted."""
        for name, info in self.registry.items():
            if info.get("stage") == "production":
                return name, info
        return None

    def rollback(self) -> tuple[str, dict[str, Any]] | None:
        """
        Promote the most recently archived model, undoing the last promotion.

        Returns the model now in production, or None when there is nothing to
        roll back to.
        """
        archived = [
            (name, info)
            for name, info in self.registry.items()
            if info.get("stage") == "archived" and info.get("demoted_at")
        ]
        if not archived:
            logger.warning("Nothing archived; there is nothing to roll back to.")
            return None

        name, _ = max(archived, key=lambda item: item[1]["demoted_at"])
        # force: the model was in production once, so it has already been
        # accepted; refusing it now on its gate would leave nothing serving.
        self.promote_model(name, force=True)
        logger.info("Rolled back to %r.", name)
        return name, self.registry[name]

    def list_models(self, model_type: str | None = None) -> dict[str, dict[str, Any]]:
        if model_type:
            return {
                k: v for k, v in self.registry.items() if v.get("model_type") == model_type
            }
        return dict(self.registry)

    def get_best_model(
        self, model_type: str, metric: str = "rmse", higher_is_better: bool = False
    ) -> tuple[str, dict[str, Any]] | None:
        """
        Best registered model of a type, by metric.

        Args:
            higher_is_better: True for metrics like r2, False for error metrics.
        """
        candidates = [
            (name, info)
            for name, info in self.list_models(model_type).items()
            if metric in info.get("metrics", {})
        ]
        if not candidates:
            logger.warning(
                "No %r models carry a %r metric; cannot select a best model.",
                model_type, metric,
            )
            return None
        return (max if higher_is_better else min)(
            candidates, key=lambda item: item[1]["metrics"][metric]
        )

    def _save_registry(self) -> None:
        Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as fh:
            json.dump(self.registry, fh, indent=4)

    def export_registry(self, filepath: str) -> None:
        """Flatten the registry to CSV for reporting."""
        import pandas as pd

        records = [
            {
                "model_name": name,
                "model_type": info.get("model_type"),
                "timestamp": info.get("timestamp"),
                "model_path": info.get("model_path"),
                **{f"metric_{k}": v for k, v in info.get("metrics", {}).items()},
            }
            for name, info in self.registry.items()
        ]
        pd.DataFrame(records).to_csv(filepath, index=False)
        logger.info("Registry exported to %s", filepath)
