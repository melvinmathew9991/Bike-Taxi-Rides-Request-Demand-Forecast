"""
`biketaxi registry`: inspect the model registry, promote a model, or roll back.

Promotion and rollback were only reachable from Python, which is not where you
want to be in the middle of an incident. After `promote` or `rollback`, tell a
running API to load the change with `POST /reload` (or restart it).

Usage:
    biketaxi registry list
    biketaxi registry promote xgb_with_lag_20261005_142518
    biketaxi registry rollback
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ML_Pipeline.registry import ModelRegistry


def _list(registry: ModelRegistry) -> int:
    rows = sorted(
        registry.list_models().items(),
        key=lambda item: item[1].get("timestamp", ""), reverse=True,
    )
    if not rows:
        print("The registry is empty.")
        return 0
    print(f"{'model':40} {'stage':12} {'gate':6} {'MASE':>6}  registered")
    for name, info in rows:
        metrics = info.get("metrics") or {}
        beats = metrics.get("beats_seasonal_naive")
        gate = "-" if beats is None else ("pass" if beats else "FAIL")
        mase = metrics.get("baseline_mase")
        # Registered but never promoted leaves stage as None, not absent.
        stage = info.get("stage") or "-"
        mase_text = "-" if mase is None else f"{mase:.4f}"
        print(
            f"{name:40} {stage:12} {gate:6} {mase_text:>6}  "
            f"{str(info.get('timestamp') or '-')[:19]}"
        )
    return 0


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog, description="Inspect the model registry, promote a model, or roll back."
    )
    parser.add_argument(
        "--output-dir", default=os.environ.get("BIKETAXI_OUTPUT_DIR", "output"),
        help="Directory holding model_registry.json (default: $BIKETAXI_OUTPUT_DIR "
             "or ./output)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Every registered model, newest first")
    promote = commands.add_parser("promote", help="Serve this model")
    promote.add_argument("name")
    promote.add_argument(
        "--force", action="store_true",
        help="Promote even if it failed, or never ran, its deploy gate",
    )
    commands.add_parser("rollback", help="Serve the previously promoted model")
    args = parser.parse_args(argv)

    registry = ModelRegistry(str(Path(args.output_dir) / "model_registry.json"))
    if args.command == "list":
        return _list(registry)

    before = registry.production_model()
    try:
        if args.command == "promote":
            registry.promote_model(args.name, force=args.force)
        else:
            if registry.rollback() is None:
                print("Nothing to roll back to: no previously promoted model.")
                return 1
    except (KeyError, ValueError) as exc:
        print(f"Refused: {exc}")
        return 1

    after = registry.production_model()
    print(
        f"Production: {before[0] if before else 'nothing'} -> "
        f"{after[0] if after else 'nothing'}. Run POST /reload, or restart the "
        "API, to serve it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
