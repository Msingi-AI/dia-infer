#!/usr/bin/env python3
"""Download msingiai/dia weights into ./models/dia."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "models" / "dia"
REPO_ID = "msingiai/dia"


def download(model_dir: Path = DEFAULT_DIR, token: str | None = None) -> Path:
    model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[download] {REPO_ID} -> {model_dir}")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(model_dir),
        token=tok,
        allow_patterns=["config.json", "model.pth", "README.md", "eval/*"],
    )
    if not (model_dir / "model.pth").exists():
        raise SystemExit(f"Download incomplete: missing {model_dir / 'model.pth'}")
    print(f"[ok] {model_dir}")
    return model_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args()
    download(args.model_dir)


if __name__ == "__main__":
    main()
