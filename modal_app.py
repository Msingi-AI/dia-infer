#!/usr/bin/env python3
"""Modal A10G serve for msingiai/dia (POST /generate → WAV, WS /tts → PCM16).

  modal deploy modal_app.py
  modal serve modal_app.py   # ephemeral

Weights download from Hugging Face (public). Optional HF_TOKEN via env if needed.
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent

app = modal.App("dia-infer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libsndfile1")
    .pip_install(
        "torch",
        "torchaudio",
        "soundfile",
        "numpy",
        "huggingface-hub",
        "descript-audio-codec",
        "pydantic",
        "fastapi",
        "uvicorn",
    )
    .env(
        {
            "DIA_MODEL_DIR": "/root/dia-infer/models/dia",
            "DIA_COMPILE": "1",
            "DIA_AUDIO_CONTEXT_TOKENS": "3072",
            "DIA_REFERENCE_MANIFEST": "/root/dia-infer/references/default.json",
        }
    )
    .add_local_dir(str(ROOT / "dia"), remote_path="/root/dia-infer/dia")
    .add_local_dir(str(ROOT / "dia_infer"), remote_path="/root/dia-infer/dia_infer")
    .add_local_dir(str(ROOT / "references"), remote_path="/root/dia-infer/references")
    .add_local_file(str(ROOT / "serve.py"), remote_path="/root/dia-infer/serve.py")
    .add_local_file(str(ROOT / "infer.py"), remote_path="/root/dia-infer/infer.py")
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    # Keep warm while sampling voices (avoids ~4 min recompile). Set 0 to scale to zero.
    min_containers=1,
)
class DiaService:
    @modal.enter()
    def setup(self) -> None:
        import os
        import sys
        from pathlib import Path

        root = Path("/root/dia-infer")
        os.chdir(root)
        sys.path.insert(0, str(root))

        model_dir = root / "models" / "dia"
        if not (model_dir / "model.pth").exists():
            from dia_infer.download import download

            token = os.environ.get("HF_TOKEN") or os.environ.get(
                "HUGGING_FACE_HUB_TOKEN"
            )
            download(model_dir, token=token or None)

        from serve import get_engine

        get_engine()
        print("[ok] DiaEngine warm on A10G")

    @modal.asgi_app()
    def fastapi_app(self):
        import sys

        sys.path.insert(0, "/root/dia-infer")
        import serve as serve_mod

        return serve_mod.app
