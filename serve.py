"""FastAPI WebSocket TTS: text in → PCM16 @ 44.1 kHz chunks out.

Production S2S uses this behind Modal A10 (local 8 GB GPUs are not realtime).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from dia_infer.engine import SAMPLE_RATE, DiaEngine
from dia_infer.stream import StreamStats

app = FastAPI(title="dia-infer")
_engine: DiaEngine | None = None


def get_engine() -> DiaEngine:
    global _engine
    if _engine is None:
        model_dir = Path(os.environ.get("DIA_MODEL_DIR", "models/dia"))
        compile = os.environ.get("DIA_COMPILE", "1") != "0"
        _engine = DiaEngine.load(model_dir, compile=compile)
    return _engine


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "sample_rate": SAMPLE_RATE,
        "loaded": _engine is not None,
        "note": "Dia S2S requires A10-class GPU; not for laptop realtime",
    }


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

        stats = StreamStats()
        for pcm in engine.stream_pcm16(text, stats=stats):
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
