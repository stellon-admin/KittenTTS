"""Kitten-ASR — speech-to-text.

Needs nothing beyond the base install; torch and transformers already ship with
the package. See docs/kittenasr.md.
"""

from kittenml import KittenASR

# Step 1: Load the model.
# Runs on GPU when one is available, otherwise on CPU.
m = KittenASR("KittenML/kitten-asr-tiny")
# m = KittenASR("./data/kitten-asr-tiny")          # local checkpoint
# m = KittenASR("KittenML/kitten-asr-tiny", device="cpu")

# Step 2: Transcribe. Accepts a file path, a URL, or a (waveform, sample_rate) tuple.
text = m.transcribe("output.wav")
print(text)

# Transcribe several files in one batch.
# print(m.transcribe(["a.wav", "b.wav"]))

# Force the output language instead of letting the model detect it.
# print(m.transcribe("output.wav", language="English"))

# Bias the transcription towards names or jargon it is likely to hear.
# print(m.transcribe("output.wav", context="Speakers: Lily, Bruno. Topic: sewing."))

# Ask for the detected language alongside the text.
result = m.transcribe_detailed("output.wav")[0]
print(f"[{result.language}] {result.text}")


# The enhanced model also reports emotion, delivery, emphasis, and accent.
enhanced = KittenASR("KittenML/kitten-asr-small-enhanced")
# enhanced = KittenASR("KittenML/kitten-asr-medium-enhanced")  # larger, same API
result = enhanced.transcribe_detailed("output.wav")[0]

print(result.text)           # markup stripped
print(result.enriched_text)  # "[Happy]Congratulations were (poured) in.[/Happy]"
print(result.accent)
print(result.emotions)       # [{'tag': 'Happy', 'segment': 'Congratulations were poured in.'}]
print(result.delivery)
print(result.word_annotations[0])

# On this model `context` is the reference transcript to condition decoding on.
# result = enhanced.transcribe_detailed("output.wav", context="Congratulations were poured in")[0]

# Serialize to the enriched-transcription JSON schema.
import json
print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
