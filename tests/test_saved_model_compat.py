"""
Models saved before the subpackage split must still load.

A pickle records its class by import path. Every bundle saved before the split
says `ML_Pipeline.features.ModelBundle` - among them the model the hosted demo
serves - and `ML_Pipeline/features.py` exists only so that path still resolves.
These tests pickle a bundle under the old path and load it back.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from joblib import load

from ML_Pipeline.modeling.features import ModelBundle

OLD = b"ML_Pipeline.features"
NEW = b"ML_Pipeline.modeling.features"


class Flat:
    def predict(self, X):
        return np.zeros(len(X))


def _pickled_under_old_path(bundle: ModelBundle) -> bytes:
    """A pickle of `bundle` that names its class the way pre-split bundles did."""
    data = pickle.dumps(bundle, protocol=4)
    # Protocol 4 writes a short module name as SHORT_BINUNICODE: opcode 0x8c,
    # one length byte, then the UTF-8 name.
    new = b"\x8c" + bytes([len(NEW)]) + NEW
    old = b"\x8c" + bytes([len(OLD)]) + OLD
    assert new in data, "pickle layout changed; rewrite this fixture"
    return data.replace(new, old)


def test_a_bundle_pickled_under_the_old_path_loads(tmp_path):
    bundle = ModelBundle(
        model=Flat(), feature_names=["lag_1"], uses_lags=True, lags=(1,),
        data_through="2021-03-26T23:30:00",
    )
    path = tmp_path / "prediction_model_with_lag_old.joblib"
    path.write_bytes(_pickled_under_old_path(bundle))
    assert OLD + b"\x94" in path.read_bytes()  # really the old path

    loaded = ModelBundle.load_bundle(path)
    assert type(loaded) is ModelBundle
    assert loaded.feature_names == ["lag_1"]
    assert loaded.data_through == "2021-03-26T23:30:00"


def test_new_bundles_record_the_new_path(tmp_path):
    path = Path(
        ModelBundle(model=Flat(), feature_names=["lag_1"], uses_lags=True).save(
            tmp_path / "m.joblib", compress=0
        )
    )
    raw = path.read_bytes()
    assert NEW in raw
    assert OLD + b"\x94" not in raw


_PROMOTED = Path("output/prediction_model_with_lag_20261005_142518.joblib")


@pytest.mark.skipif(not _PROMOTED.exists(), reason="local output directory only")
def test_the_model_the_demo_serves_still_loads():
    assert type(load(_PROMOTED)) is ModelBundle
