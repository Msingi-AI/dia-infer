#!/usr/bin/env python3
"""Local smoke CLI — write one utterance to WAV (use Modal A10 for realtime)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from dia_infer.engine import SAMPLE_RATE, DiaEngine
from dia_infer.stream import StreamStats

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "dia"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text")
    p.add_argument("-o", "--output", type=Path, default=ROOT / "out.wav")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=1.3)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--speed-factor", type=float, default=1.0)
    p.add_argument("--prompt-text", type=str, default=None)
    p.add_argument("--audio-prompt", type=Path, default=None)
    p.add_argument("--metrics", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    engine = DiaEngine.load(
        args.model_dir,
        compile=not args.no_compile,
    )
    stats = StreamStats()
    chunks = list(
        engine.stream(
            args.text,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            top_p=args.top_p,
            seed=args.seed,
            prompt_text=args.prompt_text,
            audio_prompt=args.audio_prompt,
            stats=stats,
        )
    )
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    from dia_infer.audio_post import apply_speed_factor

    audio = apply_speed_factor(audio, args.speed_factor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, SAMPLE_RATE)
    print(f"[ok] {args.output} ({len(audio) / SAMPLE_RATE:.2f}s)")
    if args.metrics:
        frames = (len(audio) / SAMPLE_RATE) * (SAMPLE_RATE / 512)
        toks = frames / max(1e-9, stats.total_s)
        print(
            json.dumps(
                {
                    "summary": stats.summary(),
                    "ttfa_s": round(stats.time_to_first_chunk_s, 4),
                    "wall_per_audio": round(stats.wall_per_audio, 3),
                    "toks_per_s": round(toks, 1),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
