"""FastAPI serve for msingiai/dia: POST /generate → WAV, WebSocket /tts → PCM16."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from dia_infer.audio_post import apply_speed_factor
from dia_infer.engine import SAMPLE_RATE, DiaEngine
from dia_infer.stream import StreamStats

app = FastAPI(title="dia-infer")
_engine: DiaEngine | None = None
_gen_lock = threading.Lock()

# Exact transcript used when generating t6.0_seed42.wav (temperature=1.3, seed=37).
T6_PROMPT_TEXT = (
    "Ninafanya kazi katika kampuni ya msingi. Nimeweza kujifunza mengi. "
    "Mnakaribishwa muweze kujaribu modeli zetu za sauti, kwa sababu hakuna "
    "modeli zingine ambazo zinafikia zetu."
)


def get_engine() -> DiaEngine:
    global _engine
    if _engine is None:
        model_dir = Path(os.environ.get("DIA_MODEL_DIR", "models/dia"))
        compile = os.environ.get("DIA_COMPILE", "1") != "0"
        _engine = DiaEngine.load(model_dir, compile=compile)
    return _engine


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    temperature: float = 1.3
    cfg_scale: float = 3.0
    top_p: float = 0.95
    seed: int | None = None
    # <1 slows (Dia clone often rushes; Gradio default try ~0.8–0.9)
    speed_factor: float = 1.0


def _wav_response(
    audio: np.ndarray,
    stats: StreamStats,
    *,
    seed: int | None,
    cloned: bool,
    speed_factor: float,
) -> Response:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    headers = {
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-Audio-Seconds": f"{len(audio) / SAMPLE_RATE:.3f}",
        "X-Wall-Seconds": f"{stats.total_s:.3f}",
        "X-Cloned": "1" if cloned else "0",
        "X-Speed-Factor": f"{speed_factor:.3f}",
    }
    if seed is not None:
        headers["X-Seed"] = str(seed)
    return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)


def _run_generate(
    *,
    text: str,
    temperature: float,
    cfg_scale: float,
    top_p: float,
    seed: int | None,
    prompt_text: str | None,
    audio_prompt_path: str | None,
    speed_factor: float = 1.0,
) -> Response:
    engine = get_engine()
    stats = StreamStats()
    with _gen_lock:
        chunks = list(
            engine.stream(
                text,
                temperature=temperature,
                cfg_scale=cfg_scale,
                top_p=top_p,
                seed=seed,
                prompt_text=prompt_text,
                audio_prompt=audio_prompt_path,
                stats=stats,
            )
        )
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    audio = apply_speed_factor(audio, speed_factor)
    return _wav_response(
        audio,
        stats,
        seed=seed,
        cloned=audio_prompt_path is not None,
        speed_factor=speed_factor,
    )


def _parse_optional_float(value: object, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)  # type: ignore[arg-type]


def _parse_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)  # type: ignore[arg-type]


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "sample_rate": SAMPLE_RATE,
        "loaded": _engine is not None,
    }


@app.post("/generate")
async def generate_wav(request: Request) -> Response:
    """Full utterance as WAV (44.1 kHz mono).

    JSON: ``{"text":"...","temperature":1.0,"seed":42}`` (no clone).

    Multipart (voice clone): fields ``text``, ``prompt_text``, file ``audio_prompt``,
    plus optional ``temperature`` / ``cfg_scale`` / ``top_p`` / ``seed``.
    ``prompt_text`` must be the exact transcript of ``audio_prompt``.
    """
    ctype = request.headers.get("content-type", "")
    tmp_path: str | None = None
    try:
        if ctype.startswith("multipart/form-data"):
            form = await request.form()
            text = str(form.get("text") or "").strip()
            if not text:
                raise HTTPException(status_code=400, detail="empty text")
            prompt_text = str(form.get("prompt_text") or "").strip() or None
            temperature = _parse_optional_float(form.get("temperature"), 1.3)
            cfg_scale = _parse_optional_float(form.get("cfg_scale"), 3.0)
            top_p = _parse_optional_float(form.get("top_p"), 0.95)
            seed = _parse_optional_int(form.get("seed"))
            speed_factor = _parse_optional_float(form.get("speed_factor"), 1.0)
            upload = form.get("audio_prompt")
            if upload is not None and hasattr(upload, "read"):
                if not prompt_text:
                    raise HTTPException(
                        status_code=400,
                        detail="prompt_text required with audio_prompt "
                        "(exact transcript of the reference wav)",
                    )
                raw = await upload.read()  # type: ignore[misc]
                if not raw:
                    raise HTTPException(status_code=400, detail="empty audio_prompt")
                name = getattr(upload, "filename", None) or "prompt.wav"
                suffix = Path(str(name)).suffix or ".wav"
                fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                os.close(fd)
                Path(tmp_path).write_bytes(raw)
            elif prompt_text:
                raise HTTPException(
                    status_code=400,
                    detail="audio_prompt file required when prompt_text is set",
                )
            return _run_generate(
                text=text,
                temperature=temperature,
                cfg_scale=cfg_scale,
                top_p=top_p,
                seed=seed,
                prompt_text=prompt_text,
                audio_prompt_path=tmp_path,
                speed_factor=speed_factor,
            )

        body = GenerateRequest.model_validate(await request.json())
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty text")
        return _run_generate(
            text=text,
            temperature=body.temperature,
            cfg_scale=body.cfg_scale,
            top_p=body.top_p,
            seed=body.seed,
            prompt_text=None,
            audio_prompt_path=None,
            speed_factor=body.speed_factor,
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.websocket("/tts")
async def tts_ws(ws: WebSocket) -> None:
    await ws.accept()
    want_metrics = ws.query_params.get("metrics", "0") in ("1", "true", "yes")
    engine = get_engine()
    try:
        raw = await ws.receive_text()
        msg = json.loads(raw)
        text = (msg.get("text") or "").strip()
        if not text:
            await ws.send_json({"event": "error", "message": "empty text"})
            await ws.close()
            return

        temperature = float(msg.get("temperature", 1.3))
        cfg_scale = float(msg.get("cfg_scale", 3.0))
        top_p = float(msg.get("top_p", 0.95))
        seed = msg.get("seed")
        seed_i = int(seed) if seed is not None else None

        stats = StreamStats()
        with _gen_lock:
            pcm_chunks = list(
                engine.stream_pcm16(
                    text,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    top_p=top_p,
                    seed=seed_i,
                    stats=stats,
                )
            )
        for pcm in pcm_chunks:
            await ws.send_bytes(pcm)

        if want_metrics and stats is not None:
            audio_s = stats.audio_seconds
            frames = audio_s * (SAMPLE_RATE / 512)
            toks = frames / max(1e-9, stats.total_s)
            await ws.send_json(
                {
                    "event": "end",
                    "sample_rate": SAMPLE_RATE,
                    "ttfa_s": round(stats.time_to_first_chunk_s, 4),
                    "wall_per_audio": round(stats.wall_per_audio, 3),
                    "toks_per_s": round(toks, 1),
                    "summary": stats.summary(),
                }
            )
        else:
            await ws.send_json({"event": "end", "sample_rate": SAMPLE_RATE})
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
