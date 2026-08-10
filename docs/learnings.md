# Inference learnings

This note records the reasoning behind the non-default inference behavior in
this repository. It is intentionally separate from the operational README.

## Reference conditioning changed the available context

The checkpoint configuration has an audio length of 1,536 DAC frames. At a
44.1 kHz sample rate and a 512-sample codec hop, that represents about 17.8
seconds of total decoder context.

The selected reference is 10.59 seconds, or approximately 912 DAC frames.
Including BOS and delay-pattern overhead left only about 608 frames—roughly
7.1 seconds—for newly generated audio. This was enough for short tests, but
longer text either truncated or was spoken unnaturally quickly.

Inference now allocates 3,072 frames. This changes runtime cache capacity only;
it does not modify checkpoint weights. With the current reference, it leaves
about 24.9 seconds for generated audio after delay overhead.

## CFG filtering

The older inference path sampled directly from classifier-free-guided logits:

```text
guided = conditional + scale × (conditional - unconditional)
```

Strong guidance can excessively sharpen or distort the distribution. Current
Dia inference instead uses the guided logits to select a top-k candidate set,
then samples the original conditional probabilities within that set. Porting
that behavior improved conditioned pacing and stability without modifying the
waveform after generation.

See the current upstream implementation in
[`dia/model.py`](https://github.com/nari-labs/dia/blob/main/dia/model.py) and the
central [Dia stability tracker](https://github.com/nari-labs/dia/issues/208).

## Early EOS

EOS used to participate in ordinary temperature and nucleus sampling. It could
therefore be selected while merely plausible rather than when the model had
actually finished. EOS is now masked unless it is the highest-logit token; when
it is highest, it is selected deterministically.

`StreamStats.stop_reason` records `eos` for a normal finish and `token_limit`
when generation consumed the allocated context.

## Text context and segmentation

The text encoder has a 512-byte context. Reference conditioning prepends the
reference transcript to every generation request, reducing the space available
for new text. Previously, excess text was silently truncated by the tokenizer.

New input is now normalized and split at punctuation or word boundaries. The
default segment budget is 220 UTF-8 bytes. Every segment uses the same cached
reference codes, keeping the voice stable while remaining within the model's
more reliable short-form range.

## Reference lifecycle

The reference audio and exact transcript live together under `references/`.
The manifest is loaded and validated when the engine starts, and the audio is
encoded into DAC codes once. Requests reuse those codes rather than uploading,
reading, and encoding the same WAV repeatedly.

This also ensures complete-WAV and streaming generation use identical voice
conditioning and sampling defaults.

## Compilation warmup

`torch.compile` specializes Python scalar generation arguments, so warming one
temperature and serving another may compile a second graph on the first real
request. The engine accepts `warmup_options`, and the service passes the same
defaults used for generation.

Warmup stops after the first decoded audio chunk. That is enough to exercise
the decoder and DAC graphs without waiting for a complete sampled utterance.
The compiled decoder wrapper is retained on the model and reused by later
streams. This avoids both unnecessary warmup audio and repeated wrapper setup;
it does not change generated audio.

The serving default remains `max-autotune`: conditioned streaming must produce
audio faster than playback to avoid buffer underruns. `reduce-overhead` starts
faster, but A10 testing produced audio at roughly 0.58x realtime. It remains
available through `DIA_COMPILE_MODE` for non-realtime comparisons.

## Seeds and voice identity

A fixed seed makes sampling reproducible for the same input. It is not a
speaker embedding, so changing the text changes the logits and sampling path.
The managed audio reference is what carries voice identity across different
sentences.

## Testing WSS streaming from a terminal

The `/tts` endpoint uses secure WebSockets (`wss://`) and returns raw signed
16-bit, little-endian, mono PCM at 44.1 kHz. The project already depends on the
Python `websockets` package, so this test does not require `websocat`:

```bash
export DIA_WSS='wss://your-modal-endpoint.modal.run/tts'
export DIA_TEXT='Habari yako leo?'

uv run python -c 'import json,os,sys; from collections import deque; from websockets.sync.client import connect; ws=connect(os.environ["DIA_WSS"]); ws.send(json.dumps({"text":os.environ["DIA_TEXT"],"temperature":1.3,"seed":37})); deque(((sys.stdout.buffer.write(message),sys.stdout.buffer.flush()) for message in ws if isinstance(message,bytes)),maxlen=0)' \
  | ffplay -nodisp -autoexit -loglevel warning \
      -f s16le -ar 44100 -ch_layout mono -i pipe:0
```

`ffplay` is installed with FFmpeg. On current FFmpeg releases, use
`-ch_layout mono`; the older `-ac 1` input option may be rejected. Do not copy
the shell prompt or a label such as `ffplay:` as part of the command.
