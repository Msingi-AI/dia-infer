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

## Install

Python 3.11 or 3.12 is required.

```bash
uv sync --python 3.11
```

Download the model files (`config.json` and `model.pth`):

```bash
uv run python -m dia_infer.download
```

## Voice reference

The default voice is defined by [`references/default.json`](references/default.json).
The audio path in a manifest is resolved relative to the manifest itself.

```json
{
  "id": "default-sw-voice",
  "language": "sw",
  "audio": "default_voice.wav",
  "transcript": "Exact transcript of the reference audio."
}
```

The transcript must match the spoken audio exactly. To use another repository
reference, add its WAV and manifest under `references/`, then pass the manifest
path to the local CLI or set `DIA_REFERENCE_MANIFEST` for the service.

## Local generation

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
from dia_infer import DiaEngine

engine = DiaEngine.load("models/dia", compile=True)

for pcm16 in engine.stream_pcm16("Habari za asubuhi."):
    send(pcm16)
```

`stream_pcm16()` yields little-endian signed 16-bit chunks. `stream()` yields
NumPy `float32` waveform chunks.

## Service

Run locally:

```bash
DIA_COMPILE=0 uv run uvicorn serve:app --host 0.0.0.0 --port 8000
```

Generate a WAV:

```bash
curl -sS http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Kila asubuhi ninaamka mapema."}' \
  -o out.wav
```

WebSocket clients connect to `/tts`, send one JSON message, then receive binary
PCM16 chunks followed by an `end` JSON event:

```json
{"text": "Habari za asubuhi.", "seed": 42}
```

## Modal

Install the host-side Modal CLI extra, then deploy:

```bash
uv sync --python 3.11 --extra modal
uv run modal deploy modal_app.py
```

The image includes the `dia/`, `dia_infer/`, and `references/` directories.
Its `dia-infer-models` Modal Volume retains model weights after the first
download.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `DIA_MODEL_DIR` | `models/dia` | Model configuration and weights directory |
| `DIA_REFERENCE_MANIFEST` | `references/default.json` | Voice-reference manifest |
| `DIA_AUDIO_CONTEXT_TOKENS` | `3072` | Runtime decoder context |
| `DIA_COMPILE` | `1` | Enable warmed `torch.compile` on CUDA |
| `DIA_COMPILE_MODE` | `reduce-overhead` | TorchInductor compilation mode |
| `DIA_TEMPERATURE` | `1.3` | Default request and warmup temperature |
| `DIA_CFG_SCALE` | `3.0` | Default request and warmup CFG scale |
| `DIA_TOP_P` | `0.95` | Default request and warmup nucleus threshold |
| `DIA_CFG_FILTER_TOP_K` | `45` | Default request and warmup CFG candidate count |
| `DIA_SEGMENT_MAX_BYTES` | `220` | Default request and warmup segment budget |

Generation requests also accept `temperature`, `cfg_scale`, `top_p`,
`cfg_filter_top_k`, `seed`, `max_tokens`, and `segment_max_bytes`.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```
