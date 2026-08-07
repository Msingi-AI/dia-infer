"""FastAPI service for WAV generation and streaming PCM16 audio."""

from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from dia_infer.engine import DEFAULT_REFERENCE_MANIFEST, SAMPLE_RATE, DiaEngine
from dia_infer.stream import DEFAULT_COMPILE_MODE, StreamStats

app = FastAPI(title="dia-infer")
_engine: DiaEngine | None = None
_gen_lock = threading.Lock()

DEFAULT_TEMPERATURE = float(os.environ.get("DIA_TEMPERATURE", "1.3"))
DEFAULT_CFG_SCALE = float(os.environ.get("DIA_CFG_SCALE", "3.0"))
DEFAULT_TOP_P = float(os.environ.get("DIA_TOP_P", "0.95"))
DEFAULT_CFG_FILTER_TOP_K = int(os.environ.get("DIA_CFG_FILTER_TOP_K", "45"))
DEFAULT_SEGMENT_MAX_BYTES = int(os.environ.get("DIA_SEGMENT_MAX_BYTES", "220"))


def _generation_defaults() -> dict[str, float | int]:
    return {
        "temperature": DEFAULT_TEMPERATURE,
        "cfg_scale": DEFAULT_CFG_SCALE,
        "top_p": DEFAULT_TOP_P,
        "cfg_filter_top_k": DEFAULT_CFG_FILTER_TOP_K,
        "segment_max_bytes": DEFAULT_SEGMENT_MAX_BYTES,
    }


def get_engine() -> DiaEngine:
    global _engine
    if _engine is None:
        model_dir = Path(os.environ.get("DIA_MODEL_DIR", "models/dia"))
        reference_manifest = Path(
            os.environ.get(
                "DIA_REFERENCE_MANIFEST", str(DEFAULT_REFERENCE_MANIFEST)
            )
        )
        _engine = DiaEngine.load(
            model_dir,
            compile=os.environ.get("DIA_COMPILE", "1") != "0",
            compile_mode=os.environ.get("DIA_COMPILE_MODE", DEFAULT_COMPILE_MODE),
            warmup_options=_generation_defaults(),
            audio_context_tokens=int(
                os.environ.get("DIA_AUDIO_CONTEXT_TOKENS", "3072")
            ),
            reference_manifest=reference_manifest,
        )
    return _engine


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0)
    cfg_scale: float = Field(DEFAULT_CFG_SCALE, ge=0.0)
    top_p: float = Field(DEFAULT_TOP_P, gt=0.0, le=1.0)
    cfg_filter_top_k: int = Field(DEFAULT_CFG_FILTER_TOP_K, gt=0)
    seed: int | None = None
    max_tokens: int | None = Field(None, gt=0)
    segment_max_bytes: int | None = Field(DEFAULT_SEGMENT_MAX_BYTES, ge=16)


def _wav_response(audio: np.ndarray, stats: StreamStats, engine: DiaEngine) -> Response:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    headers = {
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-Audio-Seconds": f"{len(audio) / SAMPLE_RATE:.3f}",
        "X-Wall-Seconds": f"{stats.total_s:.3f}",
        "X-Segments": str(stats.segments_generated),
        "X-Stop-Reason": stats.stop_reason or "unknown",
    }
    if engine.reference is not None:
        headers["X-Voice-Reference"] = engine.reference.id
    return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)


@app.get("/health")
def health() -> dict:
    engine = _engine
    return {
        "ok": True,
        "sample_rate": SAMPLE_RATE,
        "loaded": engine is not None,
        "audio_context_tokens": engine.audio_context_tokens if engine else None,
        "voice_reference": (
            engine.reference.id if engine and engine.reference is not None else None
        ),
    }


@app.post("/generate")
def generate_wav(body: GenerateRequest) -> Response:
    """Generate a complete 44.1 kHz mono WAV."""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    engine = get_engine()
    stats = StreamStats()
    with _gen_lock:
        chunks = list(
            engine.stream(
                text,
                temperature=body.temperature,
                cfg_scale=body.cfg_scale,
                top_p=body.top_p,
                cfg_filter_top_k=body.cfg_filter_top_k,
                seed=body.seed,
                max_tokens=body.max_tokens,
                segment_max_bytes=body.segment_max_bytes,
                stats=stats,
            )
        )
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return _wav_response(audio, stats, engine)


@app.websocket("/tts")
async def tts_ws(ws: WebSocket) -> None:
    await ws.accept()
    want_metrics = ws.query_params.get("metrics", "0") in ("1", "true", "yes")
    engine = get_engine()
    try:
        msg = json.loads(await ws.receive_text())
        text = str(msg.get("text") or "").strip()
        if not text:
            await ws.send_json({"event": "error", "message": "empty text"})
            await ws.close()
            return

        max_tokens_value = msg.get("max_tokens")
        segment_bytes_value = msg.get(
            "segment_max_bytes", DEFAULT_SEGMENT_MAX_BYTES
        )
        seed_value = msg.get("seed")
        stats = StreamStats()
        with _gen_lock:
            for pcm in engine.stream_pcm16(
                text,
                temperature=float(msg.get("temperature", DEFAULT_TEMPERATURE)),
                cfg_scale=float(msg.get("cfg_scale", DEFAULT_CFG_SCALE)),
                top_p=float(msg.get("top_p", DEFAULT_TOP_P)),
                cfg_filter_top_k=int(
                    msg.get("cfg_filter_top_k", DEFAULT_CFG_FILTER_TOP_K)
                ),
                max_tokens=(
                    int(max_tokens_value) if max_tokens_value is not None else None
                ),
                segment_max_bytes=(
                    int(segment_bytes_value)
                    if segment_bytes_value is not None
                    else None
                ),
                seed=int(seed_value) if seed_value is not None else None,
                stats=stats,
            ):
                await ws.send_bytes(pcm)

        if want_metrics:
            frames = stats.audio_seconds * (SAMPLE_RATE / 512)
            await ws.send_json(
                {
                    "event": "end",
                    "sample_rate": SAMPLE_RATE,
                    "ttfa_s": round(stats.time_to_first_chunk_s, 4),
                    "wall_per_audio": round(stats.wall_per_audio, 3),
                    "toks_per_s": round(frames / max(1e-9, stats.total_s), 1),
                    "summary": stats.summary(),
                }
            )
        else:
            await ws.send_json({"event": "end", "sample_rate": SAMPLE_RATE})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_json({"event": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
