"""Tests for the `biketaxi` dispatcher in `ML_Pipeline.cli`."""

from __future__ import annotations

import subprocess
import sys

import pytest
from test_serving_api import build_output_dir

from ML_Pipeline import __version__
from ML_Pipeline.cli import COMMANDS, main


def test_help_lists_every_command(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in COMMANDS:
        assert name in out


def test_no_command_prints_usage_and_exits_2(capsys):
    assert main([]) == 2
    assert "usage: biketaxi" in capsys.readouterr().out


def test_an_unknown_command_exits_2(capsys):
    assert main(["deploy"]) == 2
    assert "unknown command 'deploy'" in capsys.readouterr().err


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"biketaxi {__version__}"


def test_arguments_reach_the_subcommand(tmp_path, capsys):
    first, _ = build_output_dir(tmp_path)
    assert main(["registry", "--output-dir", str(tmp_path), "list"]) == 0
    assert first in capsys.readouterr().out


def test_subcommand_help_is_named_after_the_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["monitor", "--help"])
    assert exc.value.code == 0
    assert "usage: biketaxi monitor" in capsys.readouterr().out


@pytest.mark.parametrize("name", list(COMMANDS))
def test_every_command_module_has_a_main(name):
    import importlib

    module = importlib.import_module(COMMANDS[name][0])
    assert callable(module.main)


def test_python_dash_m_runs_the_dispatcher():
    """`python -m ML_Pipeline` works wherever the package is installed."""
    result = subprocess.run(
        [sys.executable, "-m", "ML_Pipeline", "--version"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"biketaxi {__version__}"
