"""Loading checkpoints exported in KittenML's packed low-bit format.

A quantized checkpoint ships `quantized.safetensors` and `quant_meta.json` in
place of `model.safetensors`. Quantized weights are stored as a dense bit stream
of unsigned `bits`-wide values plus per-group fp16 scales and zero points; every
other tensor is stored as-is.

The weights are unpacked back to full tensors on load, reproducing exactly what
the quantized model was evaluated with. That shrinks the download -- roughly a
quarter of the full-precision file at 4 bits -- but not the memory the model
occupies once it is running.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

QUANT_META_FILE = "quant_meta.json"
QUANT_WEIGHTS_FILE = "quantized.safetensors"


QUANTIZATIONS = ("int8", "int4")


def find_quantized_files(
    model_path: str, cache_dir=None, quantization: Optional[str] = None
) -> Optional[Tuple[str, str]]:
    """Locate a checkpoint's quantized files, downloading them if it is on the Hub.

    With `quantization` unset, looks for packed weights at the checkpoint's root
    and returns None when there are none (a normal, full-precision checkpoint).
    With `quantization="int8"` or `"int4"`, looks in that subfolder instead, so
    one repository can carry several builds next to its full-precision weights;
    a missing build is an error rather than a silent fallback.

    Returns `(meta_path, weights_path)`, or None.
    """
    if quantization is not None and quantization not in QUANTIZATIONS:
        raise ValueError(f"quantization must be one of {QUANTIZATIONS} or None")
    subfolder = quantization

    if os.path.isdir(model_path):
        folder = os.path.join(model_path, subfolder) if subfolder else model_path
        meta = os.path.join(folder, QUANT_META_FILE)
        weights = os.path.join(folder, QUANT_WEIGHTS_FILE)
        if os.path.isfile(meta) and os.path.isfile(weights):
            return meta, weights
        if subfolder:
            raise FileNotFoundError(f"{model_path!r} has no {quantization} build in {folder!r}")
        return None

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        meta = hf_hub_download(model_path, QUANT_META_FILE, subfolder=subfolder,
                               cache_dir=cache_dir)
    except EntryNotFoundError:
        if subfolder:
            raise FileNotFoundError(
                f"{model_path!r} has no {quantization} build on the Hub"
            ) from None
        return None  # a normal, unquantized repo
    return meta, hf_hub_download(model_path, QUANT_WEIGHTS_FILE, subfolder=subfolder,
                                 cache_dir=cache_dir)


def unpack_bits(packed, bits: int, n_values: int):
    """Expand a dense `bits`-per-value stream back into one uint8 per value.

    The inverse of the exporter's packing: values sit MSB-first, packed end to
    end along each row.
    """
    import torch

    if 8 % bits == 0:
        # Widths that divide a byte need only shifts, which is much faster than
        # materializing the bit stream and covers every width shipped so far.
        per_byte = 8 // bits
        shifts = torch.tensor(
            [8 - bits * (i + 1) for i in range(per_byte)], dtype=torch.uint8
        )
        values = (packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
        return values.reshape(packed.shape[0], -1)[:, :n_values].contiguous()

    import numpy as np

    array = packed.cpu().numpy()
    rows = array.shape[0]
    stream = np.unpackbits(array, axis=1)[:, : n_values * bits]
    stream = stream.reshape(rows * n_values, bits)
    pad = np.zeros((rows * n_values, 8 - bits), dtype=np.uint8)
    values = np.packbits(np.concatenate([pad, stream], axis=1), axis=1)
    return torch.from_numpy(values.reshape(rows, n_values))


def dequantize_tensor(entry: Dict[str, Any], tensors: Dict[str, Any], name: str):
    """Rebuild one weight from its packed values, scales, and zero points."""
    q = unpack_bits(tensors[name + ".q"], entry["bits"], entry["cols"]).float()
    zero = tensors[name + ".zero"].float().unsqueeze(-1)
    scale = tensors[name + ".scale"].float().unsqueeze(-1)
    grouped = q.reshape(q.shape[0], -1, entry["group"])
    return ((grouped - zero) * scale).reshape(entry["shape"])


def load_quantized_state_dict(meta_path: str, weights_path: str, dtype=None) -> Dict[str, Any]:
    """Read a quantized checkpoint into a state dict of ordinary tensors."""
    from safetensors.torch import load_file

    with open(meta_path) as handle:
        meta = json.load(handle)
    tensors = load_file(weights_path)

    state = {
        name: _cast(dequantize_tensor(entry, tensors, name), dtype)
        for name, entry in meta["tensors"].items()
    }
    for name, tensor in tensors.items():
        if not name.endswith((".q", ".scale", ".zero")):
            state[name] = _cast(tensor, dtype)
    return state


def describe_quantization(meta_path: str) -> str:
    """One-line summary of how a checkpoint was quantized."""
    with open(meta_path) as handle:
        meta = json.load(handle)
    return meta.get("plan", "unknown quantization plan")


def _cast(tensor, dtype):
    if dtype is None or not tensor.is_floating_point():
        return tensor
    return tensor.to(dtype)
