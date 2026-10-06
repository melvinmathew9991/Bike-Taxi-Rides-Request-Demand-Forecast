"""
Compatibility for saved models. Use `ML_Pipeline.modeling.features`.

A `ModelBundle` is pickled with the path of its class, and every bundle saved
before the package was split into subpackages records it as
`ML_Pipeline.features.ModelBundle` - including the model the hosted demo
serves. Without this module none of them would load. Bundles saved since
record `ML_Pipeline.modeling.features.ModelBundle` and do not need it.

`tests/test_saved_model_compat.py` loads a bundle pickled under the old path.
"""

from ML_Pipeline.modeling.features import ModelBundle

__all__ = ["ModelBundle"]
