# Kitten-ASR

Speech-to-text built on PyTorch. Needs nothing beyond the base install: torch and
transformers already come with the package.

- [Models](#models)
- [Loading a model](#loading-a-model)
- [Transcribing](#modeltranscribeaudio-context-languagenone-max_chunk_sec3000)
- [Running with vLLM](#running-with-vllm)
- [Detailed results](#modeltranscribe_detailed)
- [Long audio](#long-audio)
- [Quantized checkpoints](#quantized-checkpoints)
- [Languages](#modelsupported_languages)

## Models

| Model | Parameters | Size | Output | Download |
|---|---|---|---|---|
| kitten-asr-tiny | 470M | 900 MB | Plain text | [KittenML/kitten-asr-tiny](https://huggingface.co/KittenML/kitten-asr-tiny) |
| kitten-asr-small-enhanced | 780M | 1.5 GB | Enriched markup | [KittenML/kitten-asr-small-enhanced](https://huggingface.co/KittenML/kitten-asr-small-enhanced) |

These preview checkpoints transcribe English well, and handle French, German, Spanish, Italian,
Dutch, Polish, Portuguese, and Russian. Languages in other scripts -- Arabic, Chinese, Hindi,
Japanese, Korean -- are not usable yet.

For what the enhanced model adds, see [Enriched output](enriched-output.md).

## Loading a model

`KittenASR(model_name, cache_dir=None, device=None, dtype=None, max_new_tokens=448,
max_batch_size=8, enriched=None, backend="torch", vllm_options=None)`

Load a model from Hugging Face Hub or from a local directory.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"KittenML/kitten-asr-tiny"` | Hugging Face repository ID, or a path to a local model directory |
| `cache_dir` | `str` | `None` | Local directory for caching downloaded model files |
| `device` | `str` | `None` | Torch device. Defaults to CUDA when available, otherwise CPU |
| `dtype` | `torch.dtype` | `None` | Weight dtype. Defaults to bfloat16 on CUDA and float32 on CPU |
| `max_new_tokens` | `int` | `448` | Generation cap per audio chunk |
| `max_batch_size` | `int` | `8` | How many chunks to run through the model at once |
| `enriched` | `bool` | `None` | Whether the checkpoint emits enriched markup. Inferred from the model name by default |
| `backend` | `str` | `"torch"` | `"torch"` for PyTorch inference, or `"vllm"` for GPU inference |
| `vllm_options` | `dict` | `None` | Optional vLLM engine settings, described below |

After loading, `model.quantized` says whether the checkpoint shipped packed low-bit weights,
and `model.enriched` whether it emits markup.

## `model.transcribe(audio, context="", language=None, max_chunk_sec=300.0)`

Transcribe audio and return the text. Returns a list of strings when `audio` is a list,
otherwise a single string.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | `str \| tuple \| list` | -- | File path, URL, base64 audio string, `(waveform, sample_rate)` tuple, or a list of any of those |
| `context` | `str \| list` | `""` | Prompt biasing the transcription, e.g. names or jargon likely to appear |
| `language` | `str \| list` | `None` | Force the output language instead of detecting it |
| `max_chunk_sec` | `float` | `300.0` | Longer audio is split into chunks of at most this many seconds |

```python
model.transcribe("meeting.wav", context="Attendees: Aparna, Bruno, Kiki.")
model.transcribe(["a.wav", "b.wav"], language="English")
```

Audio is resampled to 16 kHz mono automatically, whatever the source rate or channel count.
`context` and `language` are broadcast over a batch when given as a scalar, or supplied
per-item as a list.

On the enhanced checkpoint, `context` is the reference transcript rather than a free-form
biasing prompt -- see [Enriched output](enriched-output.md).

## `model.transcribe_detailed`

Takes the same arguments as `transcribe` and always returns a list of `Transcription` objects,
one per input audio.

```python
result = model.transcribe_detailed("meeting.wav")[0]
print(result.language, result.text)
```

| Attribute | Type | Description |
|---|---|---|
| `text` | `str` | The transcribed text, markup stripped |
| `language` | `str` | Detected or forced language, e.g. `"English"` |
| `enriched_text` | `str` | Raw output with markup (enriched checkpoints only) |
| `accent` | `str \| None` | Accent the model reported (enriched only) |
| `emotions` | `list[dict]` | Emotion spans (enriched only) |
| `delivery` | `list[dict]` | Delivery spans (enriched only) |
| `word_annotations` | `list[dict]` | Per-word annotations (enriched only) |
| `reference_transcript` | `str \| None` | Transcript decoding was conditioned on (enriched only) |

`Transcription.to_dict()` serializes to the enriched-transcription JSON schema, and
`str(result)` is the text.

## `model.transcribe_file(audio_path, **kwargs)`

Alias for `transcribe` on a single file.

## Running with vLLM

Install the optional backend on Linux with a CUDA GPU:

```bash
pip install 'kittenml[vllm]'
```

```python
from kittenml import KittenASR

def main():
    model = KittenASR("KittenML/kitten-asr-tiny", backend="vllm")
    print(model.transcribe(["speech.wav", "meeting.wav"]))

if __name__ == "__main__":
    main()
```

Keep the main guard because vLLM starts a worker process. Both the tiny and enhanced
checkpoints use vLLM's native Qwen3-ASR implementation with their existing Hugging Face
weights; no checkpoint conversion is needed. `cache_dir` controls the download cache.
The same transcription methods, language/context arguments, chunking, and enriched result
format work with either backend. Decoding is greedy; small numerical differences between
engines can change the transcript.

`dtype` accepts `torch.bfloat16`, `torch.float16`, or `torch.float32`. When omitted, vLLM
uses bfloat16 if the GPU supports it, otherwise float16. Select GPUs using
`CUDA_VISIBLE_DEVICES`, leaving `device` unset or set to `"cuda"`.

Tune engine settings through `vllm_options`:

```python
model = KittenASR(
    "KittenML/kitten-asr-small-enhanced",
    backend="vllm",
    vllm_options={"gpu_memory_utilization": 0.4, "max_num_seqs": 4},
)
```

Defaults are `gpu_memory_utilization=0.25`, `max_model_len=8192`, and `max_num_seqs=8`.
Other accepted options are `tensor_parallel_size`, `enforce_eager`,
`max_num_batched_tokens`, `kv_cache_memory_bytes`, `enable_prefix_caching`, and `seed`.
`max_batch_size` controls how many audio chunks each call submits; `max_num_seqs` controls
vLLM's concurrent requests. The context window must fit the prompt, audio tokens, and
`max_new_tokens`; reduce `max_chunk_sec` or increase `max_model_len` for longer recordings.
Setting `enforce_eager=True` skips compilation and CUDA graph capture at startup.

The packed low-bit checkpoint format described below is supported by the PyTorch backend
only. The vLLM backend uses full-precision checkpoint weights.

## Long audio

Recordings longer than `max_chunk_sec` are split before inference. The split point is chosen at
the lowest-energy position within a few seconds of the target, so it tends to land in a pause
rather than mid-word, and the chunks reproduce the original audio exactly with no gaps or
overlaps. Per-chunk transcripts are concatenated on the way out.

Lower `max_chunk_sec` if you are tight on memory, and raise it to give the model more context
per chunk. For a recording that switches language between chunks, `language` comes back as a
comma-separated list in the order the languages appear.

## Quantized checkpoints

Kitten-ASR can load a checkpoint quantized to 4 bits with a group size of 128, which is
roughly a 250 MB download in place of 900 MB. No such checkpoint is published yet, so this
describes the format rather than something you can download today. Nothing changes at the
call site:

```python
model = KittenASR("KittenML/kitten-asr-tiny-int4")   # when one is published
model.transcribe("speech.wav")
```

Kitten-ASR recognizes such a checkpoint by the `quant_meta.json` and `quantized.safetensors` it
ships in place of `model.safetensors`, unpacks the weights on load, and sets `model.quantized`
to `True`.

What this buys you is **download and disk size, not memory**. The weights are expanded back to
bfloat16 (float32 on CPU) as they load, so the running model occupies exactly as much as the
unquantized one. Loading costs a few seconds more than a plain checkpoint, since the weights
have to be unpacked.

On 25 LibriSpeech test-clean samples the quantized model scored 4.21% WER against 3.82% for the
full-precision one, and produced identical text on 14 of them -- the rest differed mostly in
punctuation and proper-noun spelling. That is a small sample, indicative rather than a
benchmark.

If you need the weights to stay packed in memory as well, convert to GGUF and run under
llama.cpp; this loader is for keeping the download small.

## `model.supported_languages`

Returns the language names accepted by the `language` argument. This is the full list the
architecture understands, which is wider than what the current preview checkpoints transcribe
well -- see [Models](#models) above.
