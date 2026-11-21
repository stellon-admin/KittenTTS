# Enriched output

`KittenML/kitten-asr-small-enhanced` is trained to emit inline markup instead of plain text:

```
[Happy]Congratulations were (poured) in upon the princess.[/Happy] [Accent='other-english']
```

| Markup | Meaning |
|---|---|
| `[Tag]...[/Tag]` | An emotion or delivery span, e.g. `[Happy]`, `[Calm]`, `[Whisper]`, `[Raised_Voice]` |
| `[tag]` | A standalone non-verbal event or pause, e.g. `[laugh]`, `[sigh]`, `[Pause_medium]` |
| `(word)` | Emphasis. Nesting raises the level: `((word))` is stronger, `(((word)))` stronger still |
| `[Accent='...']` | The accent the model heard, emitted once at the end |

## Using it

Kitten-ASR detects an enriched checkpoint from its name and handles both sides of it for you --
the prompt it expects going in, the markup coming out. `text` stays clean and the annotations
arrive alongside:

```python
from kittenml import KittenASR

model = KittenASR("KittenML/kitten-asr-small-enhanced")
result = model.transcribe_detailed("speech.wav")[0]

result.text             # 'Congratulations were poured in upon the princess.'
result.enriched_text    # the raw markup above
result.accent           # 'other-english'
result.emotions         # [{'tag': 'Happy', 'segment': 'Congratulations were poured in ...'}]
result.delivery         # same shape, for delivery spans
result.word_annotations # per word: emotion, delivery, emphasis level, non-verbal event
result.to_dict()        # the full JSON schema below
```

`transcribe()` returns the plain text here too, so the enhanced model is a drop-in swap for the
plain checkpoints.

Pass `enriched=True` or `enriched=False` to `KittenASR(...)` to override the name-based
detection -- `enriched=False` leaves the raw markup in `text` and skips the special prompt.

## Reference transcripts

This checkpoint was fine-tuned behind a fixed prompt carrying a **reference transcript**, so
`context` means something different than it does on the plain models: whatever you pass is used
as that reference rather than as a free-form biasing prompt.

```python
model.transcribe_detailed("speech.wav", context="Congratulations were poured in")
```

Leave it empty and the reference is blank, which is the normal case.

## Word annotations

Every word gets an entry. Non-verbal events get their own entry with a null word, so the
sequence stays in order:

```python
{'index': 2, 'word': 'poured', 'nv-event': None,
 'emotion': 'Happy', 'delivery': 'Neutral', 'emphasis': 1}

{'index': 5, 'word': None, 'nv-event': 'laugh',
 'emotion': 'Happy', 'delivery': 'Neutral', 'emphasis': 0}
```

`emotion` and `delivery` are `'Neutral'` outside any span. `emphasis` is `0` to `3`.

## JSON schema

`result.to_dict()` produces:

```json
{
  "lang": "English",
  "accent": "other-english",
  "plain_text": "Congratulations were poured in upon the princess.",
  "reference_transcript_provided": false,
  "reference_transcript": null,
  "enrich_text": "[Happy]Congratulations were (poured) in upon the princess.[/Happy] [Accent='other-english']",
  "word_level_annotations": [],
  "emotions": [],
  "delivery": []
}
```

## Parsing markup from elsewhere

The helpers work standalone, on output produced by any means:

```python
from kittenml.kittenasr import plain_text, postprocess_enriched_text, reference_context

plain_text("[Happy]hello (there)[/Happy]")   # 'hello there'
postprocess_enriched_text(raw)               # dict of plain_text + annotations + spans
reference_context("a transcript")            # the conditioning prompt
```

The recognized tag vocabularies -- `EMOTION_TAGS`, `DELIVERY_TAGS`, `NON_VERBAL_EVENTS` -- live
in `kittenml.kittenasr.enriched`. A tag outside them is dropped from `plain_text` and ignored in
the annotations, so an unrecognized tag never corrupts the transcript.
