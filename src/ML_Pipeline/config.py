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


@dataclass
class PipelineConfig:
    """Settings for one pipeline run."""

    # --- Paths -----------------------------------------------------------
    project_root: str = ""
    data_dir: str = "data"
    raw_data_path: str = "data/raw_data.csv"
    test_data_path: str = "data/test_dataset/cleaned_test_booking_data.csv"
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

    # --- Features --------------------------------------------------------
    lag_features: tuple[int, ...] = (1, 2, 3)
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
    #: Intervals to forecast. Defaults to one day at `interval_minutes`.
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
        names = {
            "without_lag": "prediction_model_without_lag",
            "with_lag": "prediction_model_with_lag",
            "clustering": "pickup_cluster_model",
        }
        stem = names.get(model_type, model_type)
        return str(Path(self.output_dir) / f"{stem}_{self.model_version}.joblib")

    def get_data_path(self, data_type: str) -> str:
        """Path for an intermediate or output dataset."""
        names = {
            "clean": "clean_data",
            "prepared": "Data_Prepared",
            "with_lag": "data_with_lag",
            "without_lag": "data_without_lag",
        }
        stem = names.get(data_type, data_type)
        return str(Path(self.output_dir) / f"{stem}_{self.model_version}.csv")

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
            "test_fraction=%.2f, objective=%s, n_estimators=%s, "
            "early_stopping_rounds=%d",
            self.n_clusters,
            self.clustering_algorithm,
            self.freq,
            self.use_cluster_centroids,
            self.lag_features,
            self.rolling_window,
            self.test_fraction,
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
