"""
The model registry: what has been trained, how it scored, and what is serving.

`output/model_registry.json` records every trained model with its metrics,
parameters and deploy-gate verdict. Serving loads the model *promoted* here,
never simply the newest; `biketaxi registry` promotes and rolls back from the
command line.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
