"""DiaEngine — load msingiai/dia and stream audio chunks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from dia_infer.stream import StreamStats, stream_pcm16, stream_utterance
from dia_infer.text import format_clone_prompt, format_sw_prompt

SAMPLE_RATE = 44_100
_ROOT = Path(__file__).resolve().parents[1]


class DiaEngine:
    """Streaming TTS for msingiai/dia (nari@052a840 + LANG2BYTE).

    Realtime streaming needs roughly A10-class GPU with warm torch.compile.
    Laptop 8 GB GPUs are smoke-only.
    """

    def __init__(self, dia_model, *, compiled: bool = False) -> None:
        self._dia = dia_model
        self._compiled = compiled
        self.last_stats: StreamStats | None = None

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def device(self) -> torch.device:
        return self._dia.device

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device | None = None,
        dtype: str = "bfloat16",
        compile: bool = True,
        warmup_text: str = "Habari.",
    ) -> "DiaEngine":
        # Vendored nari package lives at repo root /dia
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        from dia.model import Dia

        model_dir = Path(model_dir).resolve()
        cfg = model_dir / "config.json"
        ckpt = model_dir / "model.pth"
        if not cfg.exists() or not ckpt.exists():
            raise FileNotFoundError(
                f"Missing {cfg.name}/{ckpt.name} under {model_dir}. "
                "Run: python -m dia_infer.download"
            )

        engine = Dia.from_local(
            str(cfg),
            str(ckpt),
            compute_dtype=dtype,
            device=torch.device("cpu"),
        )
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        engine.device = device
        engine.model = engine.model.to(device).eval()
        engine.dac_model = engine.dac_model.to(device)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        wrapper = cls(engine, compiled=False)
        if compile and device.type == "cuda":
            # Warm compile so first real request is fast.
            list(
                wrapper.stream(
                    warmup_text,
                    use_torch_compile=True,
                )
            )
            wrapper._compiled = True
        return wrapper

    def stream(
        self,
        text: str,
        *,
        chunk_ms: int = 250,
        cfg_scale: float = 3.0,
        temperature: float = 1.3,
        top_p: float = 0.95,
        seed: int | None = None,
        prompt_text: str | None = None,
        audio_prompt: str | Path | None = None,
        use_torch_compile: bool | None = None,
        stats: StreamStats | None = None,
    ) -> Iterator[np.ndarray]:
        if audio_prompt is not None and not (prompt_text or "").strip():
            raise ValueError("prompt_text is required when audio_prompt is set")
        if prompt_text and audio_prompt is None:
            raise ValueError("audio_prompt is required when prompt_text is set")
        if audio_prompt is not None:
            prompt = format_clone_prompt(prompt_text or "", text)
            audio_ref: str | Path | None = str(audio_prompt)
        else:
            prompt = format_sw_prompt(text)
            audio_ref = None
        use_compile = self._compiled if use_torch_compile is None else use_torch_compile
        stream_stats = stats if stats is not None else StreamStats()
        self.last_stats = stream_stats
        yield from stream_utterance(
            self._dia,
            prompt,
            chunk_ms=chunk_ms,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            audio_prompt=audio_ref,
            use_torch_compile=use_compile,
            stats=stream_stats,
        )

    def stream_pcm16(self, text: str, **kwargs) -> Iterator[bytes]:
        prompt_text = kwargs.pop("prompt_text", None)
        audio_prompt = kwargs.pop("audio_prompt", None)
        if audio_prompt is not None and not (prompt_text or "").strip():
            raise ValueError("prompt_text is required when audio_prompt is set")
        if prompt_text and audio_prompt is None:
            raise ValueError("audio_prompt is required when prompt_text is set")
        if audio_prompt is not None:
            prompt = format_clone_prompt(prompt_text or "", text)
            audio_ref: str | Path | None = str(audio_prompt)
        else:
            prompt = format_sw_prompt(text)
            audio_ref = None
        use_compile = kwargs.pop("use_torch_compile", None)
        if use_compile is None:
            use_compile = self._compiled
        stats = kwargs.pop("stats", None) or StreamStats()
        self.last_stats = stats
        yield from stream_pcm16(
            self._dia,
            prompt,
            use_torch_compile=use_compile,
            stats=stats,
            audio_prompt=audio_ref,
            **kwargs,
        )
