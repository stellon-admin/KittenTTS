import json
import os
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - allow running helper tests without torch
    torch = None

from kittenml.kittenasr.quantization import find_quantized_files

if torch is not None:
    from kittenml.kittenasr.quantization import (
        dequantize_tensor,
        describe_quantization,
        load_quantized_state_dict,
        unpack_bits,
    )


def pack_bits(values, bits):
    """The exporter's packing, reproduced here so the tests pin the format."""
    array = values.numpy().astype(np.uint8)
    rows, cols = array.shape
    stream = np.unpackbits(array.reshape(-1, 1), axis=1)[:, 8 - bits :]
    return torch.from_numpy(np.packbits(stream.reshape(rows, cols * bits), axis=1)).contiguous()


@unittest.skipIf(torch is None, "requires torch")
class UnpackBitsTests(unittest.TestCase):
    def test_round_trips_every_width(self):
        torch.manual_seed(0)
        for bits in (1, 2, 3, 4, 8):
            for cols in (16, 129, 640):
                with self.subTest(bits=bits, cols=cols):
                    values = torch.randint(0, 2**bits, (5, cols), dtype=torch.uint8)
                    unpacked = unpack_bits(pack_bits(values, bits), bits, cols)
                    self.assertTrue(torch.equal(unpacked, values))

    def test_shift_path_matches_the_bit_stream_path(self):
        # 4 bits takes the fast path, 3 bits the numpy fallback; both must agree
        # with a bit-stream expansion done by hand.
        values = torch.tensor([[0, 1, 14, 15, 7, 8]], dtype=torch.uint8)
        unpacked = unpack_bits(pack_bits(values, 4), 4, values.shape[1])
        self.assertEqual(unpacked.tolist(), values.tolist())


@unittest.skipIf(torch is None, "requires torch")
class DequantizeTests(unittest.TestCase):
    def test_applies_scale_and_zero_point_per_group(self):
        # Two groups of two: group 0 has scale 2 / zero 1, group 1 scale 10 / zero 0.
        q = torch.tensor([[3, 1, 2, 5]], dtype=torch.uint8)
        tensors = {
            "w.q": pack_bits(q, 4),
            "w.scale": torch.tensor([[2.0, 10.0]]),
            "w.zero": torch.tensor([[1.0, 0.0]]),
        }
        entry = {"bits": 4, "group": 2, "cols": 4, "shape": [1, 4]}
        out = dequantize_tensor(entry, tensors, "w")
        self.assertEqual(out.tolist(), [[4.0, 0.0, 20.0, 50.0]])

    def test_state_dict_keeps_unquantized_tensors_verbatim(self):
        with tempfile.TemporaryDirectory() as d:
            from safetensors.torch import save_file

            q = torch.tensor([[3, 1]], dtype=torch.uint8)
            save_file(
                {
                    "w.q": pack_bits(q, 4),
                    "w.scale": torch.tensor([[2.0]]),
                    "w.zero": torch.tensor([[1.0]]),
                    "bias": torch.tensor([7.0, 8.0]),
                    "counter": torch.tensor([3], dtype=torch.int64),
                },
                os.path.join(d, "quantized.safetensors"),
            )
            meta = {
                "plan": "test W4/g2",
                "tensors": {"w": {"bits": 4, "group": 2, "cols": 2, "shape": [1, 2]}},
            }
            with open(os.path.join(d, "quant_meta.json"), "w") as handle:
                json.dump(meta, handle)

            found = find_quantized_files(d)
            self.assertIsNotNone(found)
            self.assertEqual(describe_quantization(found[0]), "test W4/g2")

            state = load_quantized_state_dict(*found, dtype=torch.float32)
            self.assertEqual(sorted(state), ["bias", "counter", "w"])
            self.assertEqual(state["w"].tolist(), [[4.0, 0.0]])
            self.assertEqual(state["bias"].tolist(), [7.0, 8.0])
            # Integer tensors must not be cast to the float dtype.
            self.assertEqual(state["counter"].dtype, torch.int64)


class FindQuantizedFilesTests(unittest.TestCase):
    def test_plain_directory_is_not_quantized(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "config.json"), "w").close()
            self.assertIsNone(find_quantized_files(d))

    def test_half_a_checkpoint_is_not_quantized(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "quant_meta.json"), "w").close()
            self.assertIsNone(find_quantized_files(d))

    def test_named_build_is_read_from_its_subfolder(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "int4"))
            for name in ("quant_meta.json", "quantized.safetensors"):
                open(os.path.join(d, "int4", name), "w").close()
            self.assertIsNone(find_quantized_files(d))  # the root stays full precision
            meta, weights = find_quantized_files(d, quantization="int4")
            self.assertEqual(meta, os.path.join(d, "int4", "quant_meta.json"))
            self.assertEqual(weights, os.path.join(d, "int4", "quantized.safetensors"))
            with self.assertRaises(FileNotFoundError):
                find_quantized_files(d, quantization="int8")

    def test_unknown_build_is_rejected(self):
        with self.assertRaises(ValueError):
            find_quantized_files("KittenML/x", quantization="int2")

    def test_named_build_is_downloaded_from_the_hub_subfolder(self):
        from unittest import mock

        from huggingface_hub.utils import EntryNotFoundError

        calls = []

        def fake_download(repo, filename, subfolder=None, cache_dir=None):
            calls.append((repo, filename, subfolder, cache_dir))
            if subfolder == "int8":
                raise EntryNotFoundError("missing")
            return f"/cache/{subfolder}/{filename}"

        with mock.patch("huggingface_hub.hf_hub_download", fake_download):
            found = find_quantized_files("KittenML/m", cache_dir="/c", quantization="int4")
            self.assertEqual(found, ("/cache/int4/quant_meta.json",
                                     "/cache/int4/quantized.safetensors"))
            self.assertEqual(calls[0], ("KittenML/m", "quant_meta.json", "int4", "/c"))
            with self.assertRaises(FileNotFoundError):
                find_quantized_files("KittenML/m", quantization="int8")


@unittest.skipIf(torch is None, "requires torch")
class TiedWeightTests(unittest.TestCase):
    def test_lm_head_tied_to_embedding_may_be_stored_once(self):
        from unittest import mock

        from kittenml.kittenasr import get_model as gm

        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(4, 2)
                self.lm_head = torch.nn.Linear(2, 4, bias=False)

            def tie_weights(self):
                self.lm_head.weight = self.embed.weight

            def to(self, *args, **kwargs):
                return self

        weight = torch.arange(8.0).reshape(4, 2)
        with mock.patch("transformers.AutoConfig.from_pretrained"), \
                mock.patch("transformers.AutoModel.from_config", return_value=Tied()), \
                mock.patch.object(gm, "load_quantized_state_dict",
                                  return_value={"embed.weight": weight}):
            model = gm._load_quantized_model("m", ("meta", "w"), torch.float32, None)
        self.assertTrue(torch.equal(model.lm_head.weight, weight))

    def test_weight_genuinely_absent_still_fails(self):
        from unittest import mock

        from kittenml.kittenasr import get_model as gm

        class Untied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(2, 2, bias=False)
                self.b = torch.nn.Linear(2, 2, bias=False)

            def tie_weights(self):
                pass

            def to(self, *args, **kwargs):
                return self

        with mock.patch("transformers.AutoConfig.from_pretrained"), \
                mock.patch("transformers.AutoModel.from_config", return_value=Untied()), \
                mock.patch.object(gm, "load_quantized_state_dict",
                                  return_value={"a.weight": torch.zeros(2, 2)}):
            with self.assertRaises(RuntimeError):
                gm._load_quantized_model("m", ("meta", "w"), torch.float32, None)


if __name__ == "__main__":
    unittest.main()
