#!/usr/bin/env python3
"""Modal A10G WebSocket serve for dia-infer (required path for S2S).

Local GPUs do not meet Dia realtime; S2S should call this endpoint.

  modal deploy modal_app.py
  # → wss://<app>.modal.run/tts

  modal serve modal_app.py   # ephemeral for testing
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent

app = modal.App("dia-infer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
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
        "python-dotenv",
    )
    .env({"DIA_MODEL_DIR": "/root/dia-infer/models/dia", "DIA_COMPILE": "1"})
    .add_local_dir(str(ROOT / "dia"), remote_path="/root/dia-infer/dia")
    .add_local_dir(str(ROOT / "dia_infer"), remote_path="/root/dia-infer/dia_infer")
    .add_local_file(str(ROOT / "serve.py"), remote_path="/root/dia-infer/serve.py")
    .add_local_file(str(ROOT / "infer.py"), remote_path="/root/dia-infer/infer.py")
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("cian-hf-secret")],
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

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
        if not token:
            raise RuntimeError("cian-hf-secret missing HF_TOKEN")
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

        model_dir = root / "models" / "dia"
        if not (model_dir / "model.pth").exists():
            from dia_infer.download import download

            download(model_dir, token=token)

        # Warm load + compile (first user request stays fast).
        from serve import get_engine

        get_engine()
        print("[ok] DiaEngine warm on A10G")

    @modal.asgi_app()
    def fastapi_app(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, "/root/dia-infer")
        import serve as serve_mod

        return serve_mod.app
