import unittest

import numpy as np

from kittenml.kittenasr.audio import (
    merge_languages,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)


class LanguageTests(unittest.TestCase):
    def test_normalize_language_name(self):
        self.assertEqual(normalize_language_name("eNGLish"), "English")
        self.assertEqual(normalize_language_name("  chinese "), "Chinese")

    def test_validate_language_rejects_unknown(self):
        validate_language("English")
        with self.assertRaises(ValueError):
            validate_language("Klingon")

    def test_merge_languages_drops_repeats_and_blanks(self):
        self.assertEqual(merge_languages(["Chinese", "English", "English"]), "Chinese,English")
        self.assertEqual(merge_languages(["", "English", ""]), "English")
        self.assertEqual(merge_languages([]), "")


class ParseOutputTests(unittest.TestCase):
    def test_parses_language_and_text(self):
        self.assertEqual(
            parse_asr_output("language English<asr_text>hello there"),
            ("English", "hello there"),
        )

    def test_untagged_output_is_all_text(self):
        self.assertEqual(parse_asr_output("hello there"), ("", "hello there"))

    def test_silent_audio_yields_nothing(self):
        self.assertEqual(parse_asr_output("language None<asr_text>"), ("", ""))

    def test_forced_language_is_kept(self):
        self.assertEqual(parse_asr_output("hello", user_language="German"), ("German", "hello"))


class ChunkingTests(unittest.TestCase):
    def test_short_audio_is_a_single_chunk(self):
        wav = np.zeros(16000, dtype=np.float32)
        chunks = split_audio_into_chunks(wav, sr=16000, max_chunk_sec=30.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][1], 0.0)

    def test_chunks_cover_the_original_audio_exactly(self):
        rng = np.random.default_rng(0)
        wav = rng.standard_normal(16000 * 25).astype(np.float32)
        chunks = split_audio_into_chunks(wav, sr=16000, max_chunk_sec=10.0)

        self.assertGreater(len(chunks), 1)
        np.testing.assert_array_equal(np.concatenate([c for c, _ in chunks]), wav)

        offsets = [offset for _, offset in chunks]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(offsets[0], 0.0)


if __name__ == "__main__":
    unittest.main()
