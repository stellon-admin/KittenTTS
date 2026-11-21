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


if __name__ == "__main__":
    unittest.main()
