# dia-infer

Streaming TTS for our Swahili Dia checkpoint [`msingiai/dia`](https://huggingface.co/msingiai/dia).

## Upstream

This is **not** shown as a GitHub fork. Inference is based on:

- **Original project:** [nari-labs/dia](https://github.com/nari-labs/dia) (Dia-1.6B)
- **Vendored tree:** commit [`052a840`](https://github.com/nari-labs/dia/commit/052a840098351132d0cb533d9bd8094cbd2bf7e2) — includes [PR #163](https://github.com/nari-labs/dia/pull/163) (*Adjust KV Cache for torch.compile friendly*), which is what unlocked realtime-class speed vs the older stlohrey / pre-compile path we started from
- **Our delta:** `LANG2BYTE` (`sw=8`) for Swahili SFT prompts (`[sw]…`, never `[S1]`/`[S2]`), plus a thin `DiaEngine` / WebSocket serve

`dia/` in this repo is that nari pin + LANG2BYTE. Runtime weights are **not** in git — download from Hugging Face (`config.json` + `model.pth`).

## Reality check (serving)

| GPU | Stream realtime? |
|-----|------------------|
| Laptop RTX 5060 8 GB | **No** (~29 toks/s) — smoke only |
| Modal **A10** (warm compile) | **Yes** (~112 toks/s, TTFA ~0.46 s, wall/audio ~0.77) |

**Not** speech-to-speech. S2S either `pip install`s this package or calls `/tts`. There is no `modal_s2s.py` here.

## Install

Needs **Python 3.11 or 3.12** (not 3.13 — `descript-audio-codec` / numba).

```bash
cd dia-infer
uv sync --python 3.11
# host Modal CLI only (do not bake `modal` into a container image):
uv sync --python 3.11 --extra modal
```

- `dia/` — nari@052a840 + LANG2BYTE
- `dia_infer/` — `DiaEngine`, streamer, text normalize, download
- `serve.py` — FastAPI `/tts` WebSocket (PCM16 @ 44.1 kHz)
- `modal_app.py` — Dia-only A10 WebSocket
- `infer.py` — local wav smoke

## Engine API

```python
from dia_infer import DiaEngine

engine = DiaEngine.load("models/dia", compile=True)  # warm compile
for pcm in engine.stream_pcm16("Habari za asubuhi."):
    ...  # little-endian int16 mono @ 44100
```

## Deploy (Dia WebSocket only)

```bash
# Modal secret cian-hf-secret with HF_TOKEN
cd dia-infer
uv run modal serve modal_app.py   # or: modal deploy modal_app.py
# WebSocket: wss://<app>.modal.run/tts
```

## Local smoke

```bash
export HF_TOKEN=hf_...
python -m dia_infer.download          # pulls msingiai/dia weights
python infer.py "Habari." -o out.wav --metrics --no-compile
```
