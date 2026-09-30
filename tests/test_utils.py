"""Model name resolution."""

import pytest

from whisper_mlx import LightningWhisperMLX
from whisper_mlx.utils import QUANT_REPOS, resolve_model_path


def test_names_resolve_to_repos():
    assert resolve_model_path("turbo") == "mlx-community/whisper-turbo"
    assert resolve_model_path("tiny") == "mlx-community/whisper-tiny-mlx"


def test_repo_paths_pass_through():
    assert resolve_model_path("someone/whisper-custom") == "someone/whisper-custom"


def test_unknown_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model_path("gigantic")


@pytest.mark.parametrize("model", sorted(QUANT_REPOS))
@pytest.mark.parametrize("quant", ["4bit", "8bit"])
def test_quantized_builds_resolve(model, quant):
    assert resolve_model_path(model, quant) == QUANT_REPOS[model][quant]


def test_unknown_quant_is_rejected():
    with pytest.raises(ValueError, match="Unknown quant '3bit'"):
        resolve_model_path("tiny", "3bit")


def test_quant_for_model_without_quantized_build_is_rejected():
    with pytest.raises(ValueError, match="No 4bit build of 'turbo'"):
        resolve_model_path("turbo", "4bit")


def test_quant_with_repo_path_is_rejected():
    with pytest.raises(ValueError, match="quant only applies to model names"):
        resolve_model_path("someone/whisper-custom", "8bit")


def test_empty_quant_means_full_precision():
    assert resolve_model_path("tiny", None) == resolve_model_path("tiny", "")


def test_lightning_wrapper_rejects_unavailable_quant():
    with pytest.raises(ValueError, match="No 4bit build"):
        LightningWhisperMLX(model="turbo", quant="4bit")
