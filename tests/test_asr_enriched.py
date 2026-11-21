import unittest

from kittenml.kittenasr.enriched import (
    parse_accent,
    plain_text,
    postprocess_enriched_text,
    reference_context,
)
from kittenml.kittenasr.get_model import _is_enriched_model


class ReferenceContextTests(unittest.TestCase):
    def test_empty_reference(self):
        self.assertEqual(reference_context(), "Reference Transcript: \nEnriched Transcript:")

    def test_reference_is_trimmed(self):
        self.assertEqual(
            reference_context("  hello there  "),
            "Reference Transcript: hello there\nEnriched Transcript:",
        )


class PlainTextTests(unittest.TestCase):
    def test_strips_spans_and_emphasis(self):
        self.assertEqual(
            plain_text("[Happy]Congratulations were (poured) in.[/Happy]"),
            "Congratulations were poured in.",
        )

    def test_event_tags_become_a_space_not_a_join(self):
        self.assertEqual(
            plain_text("done. [Pause_medium] Then more."),
            "done. Then more.",
        )

    def test_punctuation_stays_attached(self):
        self.assertEqual(plain_text("Yeah [Pause_short] , right?"), "Yeah, right?")

    def test_empty_input(self):
        self.assertEqual(plain_text(""), "")


class AccentTests(unittest.TestCase):
    def test_reads_quoted_accent(self):
        self.assertEqual(parse_accent("hi. [Accent='other-english']"), "other-english")

    def test_missing_accent_is_none(self):
        self.assertIsNone(parse_accent("hi."))


class PostprocessTests(unittest.TestCase):
    def test_emotion_and_delivery_spans(self):
        out = postprocess_enriched_text("[Happy]good news[/Happy] [Whisper]quietly[/Whisper]")
        self.assertEqual(out["emotions"], [{"tag": "Happy", "segment": "good news"}])
        self.assertEqual(out["delivery"], [{"tag": "Whisper", "segment": "quietly"}])

    def test_word_carries_active_tags_and_emphasis(self):
        out = postprocess_enriched_text("[Happy]a ((b)) c[/Happy]")
        by_word = {w["word"]: w for w in out["word_level_annotations"]}
        self.assertEqual(by_word["a"]["emotion"], "Happy")
        self.assertEqual(by_word["a"]["emphasis"], 0)
        self.assertEqual(by_word["b"]["emphasis"], 2)
        self.assertEqual(by_word["c"]["emphasis"], 0)

    def test_non_verbal_event_becomes_a_wordless_annotation(self):
        out = postprocess_enriched_text("yes [laugh] no")
        events = [w for w in out["word_level_annotations"] if w["nv-event"]]
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0]["word"])
        self.assertEqual(events[0]["nv-event"], "laugh")
        self.assertEqual(out["plain_text"], "yes no")

    def test_unclosed_span_runs_to_the_end(self):
        out = postprocess_enriched_text("[Sad]it just stops")
        self.assertEqual(out["emotions"], [{"tag": "Sad", "segment": "it just stops"}])

    def test_underscored_tag_is_titlecased(self):
        out = postprocess_enriched_text("[raised_voice]hey[/raised_voice]")
        self.assertEqual(out["delivery"], [{"tag": "Raised Voice", "segment": "hey"}])

    def test_indexes_are_contiguous(self):
        out = postprocess_enriched_text("one [sigh] two three")
        self.assertEqual(
            [w["index"] for w in out["word_level_annotations"]],
            list(range(len(out["word_level_annotations"]))),
        )


class EnrichedDetectionTests(unittest.TestCase):
    def test_detects_enriched_checkpoints(self):
        self.assertTrue(_is_enriched_model("KittenML/kitten-asr-small-enhanced"))
        self.assertTrue(_is_enriched_model("/data/models/KitenASR-Enriched-0.6B/"))

    def test_plain_checkpoints_are_not_enriched(self):
        self.assertFalse(_is_enriched_model("KittenML/kitten-asr-tiny"))
        self.assertFalse(_is_enriched_model("./data/kitten-asr-mini"))


if __name__ == "__main__":
    unittest.main()
