"""Tests for scripts/registry.py, the promote and rollback command line."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from test_serving_api import add_promoted_model, build_output_dir

from ML_Pipeline.config import ModelRegistry


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "registry.py"
    spec = importlib.util.spec_from_file_location("registry_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def production(tmp_path) -> str | None:
    promoted = ModelRegistry(str(tmp_path / "model_registry.json")).production_model()
    return promoted[0] if promoted else None


def test_list_shows_every_model_and_its_stage(tmp_path, cli, capsys):
    first, _ = build_output_dir(tmp_path)
    second = add_promoted_model(tmp_path)
    assert cli(["--output-dir", str(tmp_path), "list"]) == 0
    out = capsys.readouterr().out
    assert first in out and second in out
    assert "production" in out and "archived" in out


def test_rollback_restores_the_previous_model(tmp_path, cli):
    first, _ = build_output_dir(tmp_path)
    add_promoted_model(tmp_path)
    assert cli(["--output-dir", str(tmp_path), "rollback"]) == 0
    assert production(tmp_path) == first


def test_rollback_with_nothing_to_return_to_fails(tmp_path, cli):
    build_output_dir(tmp_path)
    assert cli(["--output-dir", str(tmp_path), "rollback"]) == 1


def test_promote_refuses_a_model_that_failed_its_gate(tmp_path, cli, capsys):
    name, _ = build_output_dir(tmp_path, gate_passed=False, promote=False)
    assert cli(["--output-dir", str(tmp_path), "promote", name]) == 1
    assert "Refused" in capsys.readouterr().out
    assert production(tmp_path) is None
