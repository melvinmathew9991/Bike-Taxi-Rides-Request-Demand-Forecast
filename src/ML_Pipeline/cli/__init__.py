"""
The `biketaxi` command: one entry point for every operational task.

    biketaxi run          train, gate and optionally promote the models
    biketaxi registry     list registered models, promote, roll back
    biketaxi monitor      health check on the serving model, for a scheduler
    biketaxi stage-demo   stage aggregated-only files for a hosted demo

Installed as a console script by `pip install -e .`, and also reachable as
`python -m ML_Pipeline`. These used to be files under `scripts/` and a
`run_pipeline.py` at the root, each of which put `src/` on `sys.path` by hand;
as package modules they import like any other code and the tests import them
directly.

Each subcommand keeps its own parser and exit codes, so `biketaxi <command>
--help` documents it fully. Subcommand modules are imported only when chosen:
`run` pulls in the training stack, which `registry list` has no need for.
"""

from __future__ import annotations

import importlib
import sys

PROG = "biketaxi"

#: name -> (module, one-line summary)
COMMANDS: dict[str, tuple[str, str]] = {
    "run": ("ML_Pipeline.cli.run", "train, gate and optionally promote the models"),
    "registry": ("ML_Pipeline.cli.registry", "list registered models, promote, roll back"),
    "monitor": ("ML_Pipeline.cli.monitor", "health check on the serving model"),
    "stage-demo": ("ML_Pipeline.cli.stage_demo", "stage aggregated-only files for a hosted demo"),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [f"usage: {PROG} <command> [options]", "", "commands:"]
    lines += [f"  {name:{width}}  {summary}" for name, (_, summary) in COMMANDS.items()]
    lines += ["", f"Run '{PROG} <command> --help' for a command's options."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return 0 if args else 2
    if args[0] in {"-V", "--version"}:
        from ML_Pipeline import __version__

        print(f"{PROG} {__version__}")
        return 0

    name, rest = args[0], args[1:]
    if name not in COMMANDS:
        print(f"{PROG}: unknown command {name!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[name][0])
    return module.main(rest, prog=f"{PROG} {name}")
