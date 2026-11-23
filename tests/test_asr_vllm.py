"""vLLM routing and multimodal request tests; no GPU or downloads required."""

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from kittenml import KittenASR
from kittenml.kittenasr import vllm as backend


@pytest.fixture
def fake_vllm(monkeypatch):
    monkeypatch.setattr(backend, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(backend, "find_quantized_files", Mock(return_value=None))
    module = ModuleType("vllm")
    module.LLM = Mock()
    module.envs = SimpleNamespace(disable_envs_cache=Mock())
    module.SamplingParams = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    processor_module = ModuleType("vllm.transformers_utils.processors.qwen3_asr")
    processor_module.Qwen3ASRProcessor = Mock()
    processor = processor_module.Qwen3ASRProcessor.from_pretrained.return_value
    processor.apply_chat_template.side_effect = lambda messages, **kw: (
        messages[0]["content"] + "<|audio_start|><|audio_pad|><|audio_end|>"
    )
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(sys.modules, processor_module.__name__, processor_module)
    return module


def output(text):
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


def test_constructor_routes_without_loading_torch_model(fake_vllm, monkeypatch):
    import kittenml.kittenasr.auto as auto
    register = Mock(side_effect=AssertionError("must not load Transformers architecture"))
    monkeypatch.setattr(auto, "register_auto_classes", register)
    model = KittenASR("kitten-asr-tiny", backend="vllm", cache_dir="/cache",
                      dtype=torch.float16, vllm_options={"max_model_len": 4096})
    register.assert_not_called()
    args = fake_vllm.LLM.call_args.kwargs
    assert args["model"] == "KittenML/kitten-asr-tiny"
    assert args["download_dir"] == "/cache"
    assert args["dtype"] == "float16"
    assert args["max_model_len"] == 4096
    assert args["hf_overrides"] == {
        "model_type": "qwen3_asr", "architectures": ["Qwen3ASRForConditionalGeneration"]
    }
    assert args["limit_mm_per_prompt"] == {"audio": 1}
    assert model.model is fake_vllm.LLM.return_value
    backend.find_quantized_files.assert_called_once_with("KittenML/kitten-asr-tiny", cache_dir="/cache")


def test_batch_requests_preserve_audio_prompts_and_order(fake_vllm):
    model = KittenASR(backend="vllm", max_batch_size=1, max_new_tokens=77)
    model.model.generate.side_effect = [[output("Hello.")], [output("Bonjour.")]]
    wav = np.ones(16000, dtype=np.float32)
    results = model.transcribe_detailed(
        [(wav, 16000), (wav * 0.5, 16000)], context=["Names", "Noms"],
        language=["english", "french"],
    )
    assert [(r.text, r.language) for r in results] == [("Hello.", "English"), ("Bonjour.", "French")]
    first, second = model.model.generate.call_args_list
    request = first.args[0][0]
    assert request["prompt"].startswith("Names")
    assert request["prompt"].endswith("language English<asr_text>")
    samples, sr = request["multi_modal_data"]["audio"]
    np.testing.assert_array_equal(samples, wav)
    assert sr == 16000
    assert second.args[0][0]["prompt"].endswith("language French<asr_text>")
    assert first.kwargs["sampling_params"].temperature == 0.0
    assert first.kwargs["sampling_params"].max_tokens == 77


def test_enriched_prompt_and_result(fake_vllm):
    model = KittenASR("kitten-asr-small-enhanced", backend="vllm")
    model.model.generate.return_value = [output("[Happy]Hello.[/Happy] [Accent='british']")]
    result = model.transcribe_detailed((np.zeros(16000), 16000), context="Hello.")[0]
    assert result.text == "Hello."
    assert result.accent == "british"
    assert result.emotions == [{"tag": "Happy", "segment": "Hello."}]
    prompt = model.model.generate.call_args.args[0][0]["prompt"]
    assert "Reference Transcript: Hello.\nEnriched Transcript:" in prompt


@pytest.mark.parametrize("kwargs, message", [
    ({"backend": "other"}, "backend must"),
    ({"vllm_options": {}}, "requires backend"),
    ({"backend": "vllm", "vllm_options": {"model": "other"}}, "unknown ASR"),
    ({"backend": "vllm", "device": "cpu"}, "CUDA GPU"),
    ({"backend": "vllm", "device": "cuda:1"}, "CUDA_VISIBLE_DEVICES"),
    ({"backend": "vllm", "dtype": "float16"}, "dtype must"),
])
def test_invalid_options_fail_before_engine_creation(fake_vllm, kwargs, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        KittenASR(**kwargs)
    fake_vllm.LLM.assert_not_called()


def test_packed_weights_are_rejected(fake_vllm):
    backend.find_quantized_files.return_value = ("meta.json", "weights.safetensors")
    with pytest.raises(ValueError, match="Packed Kitten-ASR"):
        KittenASR(backend="vllm")
    fake_vllm.LLM.assert_not_called()


def test_missing_optional_dependency(fake_vllm, monkeypatch):
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match=r"kittenml\[vllm\]"):
        KittenASR(backend="vllm")


@pytest.mark.parametrize("previous", [None, "1"])
def test_sampler_default_is_scoped_even_when_engine_fails(fake_vllm, monkeypatch, previous):
    key = "VLLM_USE_FLASHINFER_SAMPLER"
    if previous is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, previous)

    def fail(**kwargs):
        assert os.environ[key] == (previous or "0")
        raise RuntimeError("engine startup failed")

    fake_vllm.LLM.side_effect = fail
    with pytest.raises(RuntimeError, match="engine startup failed"):
        KittenASR(backend="vllm")
    assert os.environ.get(key) == previous
    fake_vllm.envs.disable_envs_cache.assert_called_once()
