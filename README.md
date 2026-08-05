# dia-infer

Streaming TTS for [`msingiai/dia`](https://huggingface.co/msingiai/dia) on **nari-labs/dia @ `052a840`** with `LANG2BYTE` (`sw=8`) baked in.

## Reality check (serving)

| GPU | Stream realtime? |
|-----|------------------|
| Laptop RTX 5060 8 GB | **No** (~29 toks/s) — smoke only |
| Modal **A10** (warm compile) | **Yes** (~112 toks/s, TTFA ~0.46 s, wall/audio ~0.77) |

**Speech-to-speech must use the Modal A10 WebSocket**, not in-process local Dia. Local S2S + Dia will not keep up.

## Install

Needs **Python 3.11 or 3.12** (not 3.13 — `descript-audio-codec` / numba).

```bash
cd dia-infer
uv sync --python 3.11
# modal CLI: uv run modal …
# or: source .venv/bin/activate && modal …
```

- `dia/` — vendored nari pin + LANG2BYTE
- `dia_infer/` — `DiaEngine`, streamer, text normalize, download
- `serve.py` — FastAPI `/tts` WebSocket (PCM16 @ 44.1 kHz)
- `modal_app.py` — A10G deploy (`min_containers=1`)
- `infer.py` — local wav smoke only

## Engine API

```python
from dia_infer import DiaEngine

engine = DiaEngine.load("models/dia", compile=True)  # warm compile
for pcm in engine.stream_pcm16("Habari za asubuhi."):
    ...  # little-endian int16 mono @ 44100
```

## Deploy (S2S path)

```bash
# needs Modal secret cian-hf-secret with HF_TOKEN
cd dia-infer
modal deploy modal_app.py
# WebSocket: wss://<your-app>.modal.run/tts
# Client sends: {"text": "Habari."}
# Server sends: binary PCM16 frames, then {"event":"end","sample_rate":44100}
```

## Local 

```bash
export HF_TOKEN=hf_...
python -m dia_infer.download
python infer.py "Habari." -o out.wav --metrics --no-compile   # laptop
```


