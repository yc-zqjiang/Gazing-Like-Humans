"""Smoke tests that exercise the public API without loading checkpoints
or downloading the DINOv2 backbone.

These tests verify that the package imports cleanly and that the model factory
exposes every advertised alias. They are intentionally lightweight so they can
run in CI on a CPU runner without GPU or network access.
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "gazelle",
        "gazelle.backbone",
        "gazelle.dataloader",
        "gazelle.model",
        "gazelle.utils",
    ],
)
def test_package_imports(module):
    importlib.import_module(module)


def test_factory_has_public_aliases():
    from gazelle.model import get_gazelle_model

    # `get_gazelle_model` builds a fresh model per call; we can't easily
    # exercise it here without DINOv2 + downloads, so we inspect the source
    # to confirm the public aliases are registered.
    import inspect

    src = inspect.getsource(get_gazelle_model)
    for alias in (
        "GazeFollow_glh_vitb14",
        "GazeFollow_glh_vitl14",
        "VAT_glh_vitb14",
        "VAT_glh_vitl14",
    ):
        assert alias in src, f"factory is missing public alias: {alias}"


def test_invalid_model_raises():
    from gazelle.model import get_gazelle_model

    with pytest.raises(AssertionError):
        get_gazelle_model("nonexistent_factory_name")
