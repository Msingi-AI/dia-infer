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

from dia.generation import DEFAULT_GENERATION_CONFIG, GenerationConfig
from dia_infer.engine import (
    DEFAULT_AUDIO_CONTEXT_TOKENS,
    DEFAULT_REFERENCE_MANIFEST,
    SAMPLE_RATE,
    DiaEngine,
)
from dia_infer.stream import DEFAULT_COMPILE_MODE, StreamStats

app = FastAPI(title="dia-infer")
_engine: DiaEngine | None = None
_gen_lock = threading.Lock()

SERVICE_GENERATION_CONFIG = GenerationConfig.from_env()


def _generation_defaults() -> dict[str, object]:
    return SERVICE_GENERATION_CONFIG.as_warmup_options()


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
                os.environ.get(
                    "DIA_AUDIO_CONTEXT_TOKENS",
                    str(DEFAULT_AUDIO_CONTEXT_TOKENS),
                )
            ),
            reference_manifest=reference_manifest,
        )
    return _engine


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    temperature: float = Field(SERVICE_GENERATION_CONFIG.temperature, ge=0.0)
    cfg_scale: float = Field(SERVICE_GENERATION_CONFIG.cfg_scale, ge=0.0)
    top_p: float = Field(SERVICE_GENERATION_CONFIG.top_p, gt=0.0, le=1.0)
    cfg_filter_top_k: int = Field(
        SERVICE_GENERATION_CONFIG.cfg_filter_top_k, gt=0
    )
    seed: int | None = SERVICE_GENERATION_CONFIG.seed
    max_tokens: int | None = Field(SERVICE_GENERATION_CONFIG.max_tokens, gt=0)
    segment_max_bytes: int | None = Field(
        SERVICE_GENERATION_CONFIG.segment_max_bytes, ge=16
    )
    chunk_ms: int = Field(SERVICE_GENERATION_CONFIG.chunk_ms, gt=0)

    def generation_config(self) -> GenerationConfig:
        return GenerationConfig.from_mapping(
            self.model_dump(exclude={"text"}),
            base=DEFAULT_GENERATION_CONFIG,
        )


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
                generation_config=body.generation_config(),
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

        generation = GenerationConfig.from_mapping(
            msg,
            base=SERVICE_GENERATION_CONFIG,
        )
        stats = StreamStats()
        with _gen_lock:
            for pcm in engine.stream_pcm16(
                text,
                generation_config=generation,
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
