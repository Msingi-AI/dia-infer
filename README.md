# dia-infer

Streaming TTS for our Swahili Dia checkpoint [`msingiai/dia`](https://huggingface.co/msingiai/dia).

## Upstream

This is **not** shown as a GitHub fork. Inference is based on:

- **Original project:** [Dia stlohrey fork](https://github.com/stlohrey/dia-finetuning)
- **Vendored tree:** commit [`052a840`](https://github.com/nari-labs/dia/commit/052a840098351132d0cb533d9bd8094cbd2bf7e2) — includes [PR #163](https://github.com/nari-labs/dia/pull/163) (*Adjust KV Cache for torch.compile friendly*), which is what unlocked realtime-class speed vs the older stlohrey / pre-compile path we started from
- **Our delta:** `LANG2BYTE` (`sw=8`) for Swahili SFT prompts (`[sw]…`, never `[S1]`/`[S2]`), plus a thin `DiaEngine` / WebSocket serve

`dia/` in this repo is that nari pin + LANG2BYTE. Runtime weights are **not** in git — download from Hugging Face (`config.json` + `model.pth`).

## Reality check (serving)

| GPU | Stream realtime? |
|-----|------------------|
| Laptop RTX 5060 8 GB | **No** (~29 toks/s) — smoke only |
| Modal **A10** (warm compile) | **Yes** (~112 toks/s, TTFA ~0.46 s, wall/audio ~0.77) |


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

## Deploy (Modal A10)

```bash
cd dia-infer
uv run modal serve modal_app.py   # ephemeral, or: modal deploy modal_app.py
# HTTPS:  https://<app>--diaservice-fastapi-app.modal.run
# WS:     wss://<app>--diaservice-fastapi-app.modal.run/tts
```

Cold start includes `torch.compile` (~few minutes). `min_containers=1` keeps the GPU warm while you sample.

### Listen / sample (POST → WAV)

```bash
URL=https://<app>--diaservice-fastapi-app.modal.run

# no clone (voice varies by text — expected)
curl -sS -X POST "$URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Habari za asubuhi.","temperature":1.0,"seed":42}' \
  -o out.wav

# voice clone from a reference wav (e.g. your liked t6 male clip)
# prompt_text MUST match the spoken words in audio_prompt exactly
PROMPT='Ninafanya kazi katika kampuni ya msingi. Nimeweza kujifunza mengi. Mnakaribishwa muweze kujaribu modeli zetu za sauti, kwa sababu hakuna modeli zingine ambazo zinafikia zetu.'

curl -sS -X POST "$URL/generate" \
  -F "text=Kila asubuhi ninaamka mapema." \
  -F "prompt_text=$PROMPT" \
  -F "audio_prompt=@t6.0_seed42.wav" \
  -F "temperature=1.0" \
  -F "seed=42" \
  -F "speed_factor=0.85" \
  -o cloned.wav
```

Clone often rushes (known Dia CFG quirk — see [nari#139](https://github.com/nari-labs/dia/issues/139)). `speed_factor<1` slows after generation (Gradio-style); try `0.8`–`0.9`. Simple stretch can shift pitch slightly.

Seed alone does **not** keep the same speaker across different sentences. Clone with `audio_prompt` + matching `prompt_text` (nari: transcript of the prompt audio **before** the new text; for `msingiai/dia` that is `[sw]…` + `[sw]…`). Output is only the new `text`, not the prompt audio. Prefer ~5–10 s of clear reference audio.

Streaming PCM remains at `wss://…/tts` (JSON `{"text":"…"}` then binary chunks).

## Local smoke

```bash
python -m dia_infer.download          # pulls msingiai/dia (public; HF_TOKEN optional)
python infer.py "Habari." -o out.wav --metrics --no-compile
```
