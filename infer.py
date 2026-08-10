#!/usr/bin/env python3
"""Local smoke CLI — write one utterance to WAV (use Modal A10 for realtime)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from dia.generation import DEFAULT_GENERATION_CONFIG, GenerationConfig
from dia_infer.engine import (
    DEFAULT_AUDIO_CONTEXT_TOKENS,
    DEFAULT_REFERENCE_MANIFEST,
    SAMPLE_RATE,
    DiaEngine,
)
from dia_infer.stream import DEFAULT_COMPILE_MODE, StreamStats

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "dia"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text")
    p.add_argument("-o", "--output", type=Path, default=ROOT / "out.wav")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--temperature", type=float, default=DEFAULT_GENERATION_CONFIG.temperature
    )
    p.add_argument(
        "--cfg-scale", type=float, default=DEFAULT_GENERATION_CONFIG.cfg_scale
    )
    p.add_argument(
        "--top-p", type=float, default=DEFAULT_GENERATION_CONFIG.top_p
    )
    p.add_argument(
        "--cfg-filter-top-k",
        type=int,
        default=DEFAULT_GENERATION_CONFIG.cfg_filter_top_k,
    )
    p.add_argument("--seed", type=int, default=DEFAULT_GENERATION_CONFIG.seed)
    p.add_argument(
        "--audio-context-tokens", type=int, default=DEFAULT_AUDIO_CONTEXT_TOKENS
    )
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_GENERATION_CONFIG.max_tokens
    )
    p.add_argument(
        "--segment-max-bytes",
        type=int,
        default=DEFAULT_GENERATION_CONFIG.segment_max_bytes,
    )
    p.add_argument(
        "--chunk-ms", type=int, default=DEFAULT_GENERATION_CONFIG.chunk_ms
    )
    p.add_argument(
        "--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST
    )
    p.add_argument("--metrics", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--compile-mode", default=DEFAULT_COMPILE_MODE)
    args = p.parse_args()

    generation = GenerationConfig(
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        top_p=args.top_p,
        cfg_filter_top_k=args.cfg_filter_top_k,
        seed=args.seed,
        max_tokens=args.max_tokens,
        segment_max_bytes=args.segment_max_bytes,
        chunk_ms=args.chunk_ms,
    )

    engine = DiaEngine.load(
        args.model_dir,
        compile=not args.no_compile,
        compile_mode=args.compile_mode,
        warmup_options=generation.as_warmup_options(),
        audio_context_tokens=args.audio_context_tokens,
        reference_manifest=args.reference_manifest,
    )
    stats = StreamStats()
    chunks = list(
        engine.stream(
            args.text,
            generation_config=generation,
            stats=stats,
        )
    )
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
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
