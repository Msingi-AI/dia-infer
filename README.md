# dia-infer

Streaming inference for the Swahili Dia checkpoint
[`msingiai/dia`](https://huggingface.co/msingiai/dia).

The service uses a repository-managed voice reference by default and produces
mono PCM/WAV audio at 44.1 kHz. Long input is segmented automatically while
retaining the same reference voice.

## Repository layout

- `dia/` — vendored Dia inference code with Swahili `LANG2BYTE` support.
- `dia_infer/` — model loading, text handling, reference loading, and streaming.
- `references/` — voice-reference audio and manifest.
- `infer.py` — local WAV generation.
- `serve.py` — FastAPI HTTP and WebSocket service.
- `modal_app.py` — Modal A10 deployment.

The vendored code is based on Nari commit
[`052a840`](https://github.com/nari-labs/dia/commit/052a840098351132d0cb533d9bd8094cbd2bf7e2)
with the Swahili tag mapped to byte `8`.

## Quick start on Modal

Modal is the recommended way to test this repository. It provides the A10 GPU
needed for realtime generation, downloads the model automatically, and retains
the weights in the `dia-infer-models` Volume.

Python 3.11 or 3.12 and a Modal account are required. Install the project and
authenticate the Modal CLI:

```bash
uv sync --python 3.11 --extra modal
uv run modal setup
```

Start an ephemeral development endpoint:

```bash
uv run modal serve modal_app.py
```

Modal prints an HTTPS URL ending in `.modal.run`. Keep `modal serve` running,
then use that URL from another terminal:

```bash
export DIA_URL='https://your-modal-endpoint.modal.run'

curl -fsS "$DIA_URL/health"

curl -fsS "$DIA_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Kila asubuhi ninaamka mapema.",
    "temperature": 1.3,
    "cfg_scale": 3.0,
    "top_p": 0.95,
    "cfg_filter_top_k": 45,
    "seed": 37,
    "segment_max_bytes": 220,
    "chunk_ms": 250
  }' \
  -o modal-output.wav
```

The same deployment exposes streaming PCM16 over WebSocket. Replace `https`
with `wss` and connect to the `/tts` path, for example
`wss://your-modal-endpoint.modal.run/tts`. Send the same JSON fields shown
above; the socket returns binary chunks followed by an `end` JSON event.

Deploy a persistent endpoint when development is complete:

```bash
uv run modal deploy modal_app.py
```

The first container compiles Dia before becoming ready. Later requests reuse
the warm container; model downloads are also reused across containers.

## Voice reference

The default voice is defined by [`references/default.json`](references/default.json).
The audio path in a manifest is resolved relative to the manifest itself.

```json
{
  "id": "t6",
  "language": "sw",
  "voices": {
    "t6": {
      "audio": "default_voice.wav",
      "transcript": "Exact transcript of the t6 audio."
    },
    "t4": {
      "audio": "t4_voice.wav",
      "transcript": "Exact transcript of the t4 audio."
    }
  }
}
```

The transcript must match its audio exactly. Select a repository voice by
changing only the top-level `id`.

## Optional local generation

Local inference is mainly a smoke test unless the machine has an A10-class GPU.
Install dependencies and download `config.json` plus `model.pth`:

```bash
uv sync --python 3.11
uv run python -m dia_infer.download
```

```bash
uv run python infer.py \
  "Kila asubuhi ninaamka mapema." \
  --output out.wav \
  --metrics \
  --no-compile
```

The default reference is loaded automatically. An explicit manifest is only
needed when selecting a different voice:

```bash
uv run python infer.py "Habari." \
  --reference-manifest references/another_voice.json \
  --output out.wav
```

## Python API

```python
from dia_infer import DiaEngine, GenerationConfig

engine = DiaEngine.load("models/dia", compile=True)
generation = GenerationConfig(temperature=1.3, seed=37)

for pcm16 in engine.stream_pcm16(
    "Habari za asubuhi.",
    generation_config=generation,
):
    send(pcm16)
```

`stream_pcm16()` yields little-endian signed 16-bit chunks. `stream()` yields
NumPy `float32` waveform chunks.

## Optional local service

Run locally:

```bash
DIA_COMPILE=0 uv run uvicorn serve:app --host 0.0.0.0 --port 8000
```

Generate a WAV:

```bash
curl -sS http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Habari rafiki yangu, karibu kwenye Sauti TTS","temperature":1.3,"seed":37}' \
  -o out.wav
```

WebSocket clients connect to `/tts`, send one JSON message, then receive binary
PCM16 chunks followed by an `end` JSON event:

```json
{"text":"Habari za asubuhi.","temperature":1.3,"seed":37}
```

## Configuration

All generation defaults live in
[`dia/generation.py`](dia/generation.py). For example, changing `temperature`
there is enough to update the model, engine, streaming path, CLI, HTTP service,
WebSocket service, and compile warmup.

For one call, override fields through CLI flags, a `GenerationConfig`, or the
HTTP/WebSocket JSON body. For a deployed service, the following environment
variables override the central defaults without editing code.

| Variable | Default | Purpose |
|---|---:|---|
| `DIA_MODEL_DIR` | `models/dia` | Model configuration and weights directory |
| `DIA_REFERENCE_MANIFEST` | `references/default.json` | Voice-reference manifest |
| `DIA_AUDIO_CONTEXT_TOKENS` | `3072` | Runtime decoder context |
| `DIA_COMPILE` | `1` | Enable warmed `torch.compile` on CUDA |
| `DIA_COMPILE_MODE` | `max-autotune` | TorchInductor compilation mode |
| `DIA_TEMPERATURE` | `1.3` | Default request and warmup temperature |
| `DIA_CFG_SCALE` | `3.0` | Default request and warmup CFG scale |
| `DIA_TOP_P` | `0.95` | Default request and warmup nucleus threshold |
| `DIA_CFG_FILTER_TOP_K` | `45` | Default request and warmup CFG candidate count |
| `DIA_SEED` | unset | Default deterministic seed |
| `DIA_MAX_TOKENS` | unset | Optional generation-frame limit |
| `DIA_SEGMENT_MAX_BYTES` | `220` | Default request and warmup segment budget |
| `DIA_CHUNK_MS` | `250` | Streaming audio chunk target |

Generation requests also accept `temperature`, `cfg_scale`, `top_p`,
`cfg_filter_top_k`, `seed`, `max_tokens`, `segment_max_bytes`, and `chunk_ms`.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```
