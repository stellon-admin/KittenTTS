"""Optional vLLM inference using its native Qwen3-ASR architecture.

Kitten-ASR checkpoints retain the Qwen3-ASR tensor layout. Config overrides
select that implementation without exporting or modifying the checkpoint.
"""

import importlib.util
import os
import sys
from contextlib import contextmanager

from .audio import SAMPLE_RATE
from .quantization import find_quantized_files


_DEFAULTS = {
    "gpu_memory_utilization": 0.25,
    "max_model_len": 8192,
    "max_num_seqs": 8,
}
_OPTIONS = set(_DEFAULTS) | {
    "tensor_parallel_size", "enforce_eager", "max_num_batched_tokens",
    "kv_cache_memory_bytes", "enable_prefix_caching", "seed",
}
_HF_OVERRIDES = {
    "model_type": "qwen3_asr",
    "architectures": ["Qwen3ASRForConditionalGeneration"],
}


@contextmanager
def _engine_environment():
    # Greedy ASR decoding does not need FlashInfer's optional sampler JIT.
    # Like the TTS backend, default to native sampling so loading a model does
    # not require a separate, matching CUDA toolkit. Respect explicit overrides.
    key = "VLLM_USE_FLASHINFER_SAMPLER"
    previous = os.environ.get(key)
    os.environ.setdefault(key, "0")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


class VLLMBackend:
    def __init__(self, model_path, cache_dir=None, device=None, dtype=None, options=None):
        import torch

        options = dict(options or {})
        unknown = set(options) - _OPTIONS
        if unknown:
            raise ValueError(f"unknown ASR vllm_options: {sorted(unknown)}")
        if sys.platform != "linux":
            raise RuntimeError("Kitten-ASR's vLLM backend requires Linux and a CUDA GPU")
        if importlib.util.find_spec("vllm") is None:
            raise ImportError('Install the optional backend with pip install "kittenml[vllm]"')
        self.device = torch.device(device or "cuda")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Kitten-ASR's vLLM backend requires a CUDA GPU")
        if self.device.index not in (None, 0):
            raise ValueError("Select GPUs with CUDA_VISIBLE_DEVICES and use device='cuda'")
        if dtype not in (None, torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("dtype must be torch.float16, torch.bfloat16, or torch.float32")
        self.dtype = dtype or (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        if find_quantized_files(model_path, cache_dir=cache_dir) is not None:
            raise ValueError("Packed Kitten-ASR checkpoints require backend='torch'; "
                             "use a full-precision checkpoint with vLLM")

        from vllm import LLM
        from vllm.transformers_utils.processors.qwen3_asr import Qwen3ASRProcessor

        with _engine_environment():
            from vllm import envs
            envs.disable_envs_cache()
            self.engine = LLM(
                model=model_path,
                download_dir=cache_dir,
                dtype=str(self.dtype).removeprefix("torch."),
                hf_overrides=dict(_HF_OVERRIDES),
                limit_mm_per_prompt={"audio": 1},
                generation_config="vllm",
                **{**_DEFAULTS, **options},
            )
        # vLLM's config loading registers the checkpoint's model_type alias.
        # Use its processor explicitly so no Transformers ASR model is loaded.
        self.processor = Qwen3ASRProcessor.from_pretrained(
            model_path, cache_dir=cache_dir, fix_mistral_regex=True,
        )

    def generate(self, prompts, wavs, max_new_tokens, batch_size):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = []
        for start in range(0, len(prompts), batch_size):
            requests = [
                {"prompt": prompt, "multi_modal_data": {"audio": (wav, SAMPLE_RATE)}}
                for prompt, wav in zip(
                    prompts[start:start + batch_size], wavs[start:start + batch_size]
                )
            ]
            results = self.engine.generate(requests, sampling_params=params, use_tqdm=False)
            outputs.extend(result.outputs[0].text for result in results)
        return outputs
