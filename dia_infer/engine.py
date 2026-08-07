"""DiaEngine — load msingiai/dia and stream audio chunks."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from dia_infer.reference import VoiceReference
from dia_infer.stream import StreamStats, stream_utterance
from dia_infer.text import format_clone_prompt, format_sw_prompt, split_for_tts

SAMPLE_RATE = 44_100
DEFAULT_AUDIO_CONTEXT_TOKENS = 3_072
DEFAULT_SEGMENT_MAX_BYTES = 220
_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_MANIFEST = _ROOT / "references" / "default.json"


class DiaEngine:
    """Streaming TTS for msingiai/dia (nari@052a840 + LANG2BYTE).

    Realtime streaming needs roughly A10-class GPU with warm torch.compile.
    Laptop 8 GB GPUs are smoke-only.
    """

    def __init__(
        self,
        dia_model,
        *,
        reference: VoiceReference | None = None,
        reference_codes: torch.Tensor | None = None,
        compiled: bool = False,
    ) -> None:
        if (reference is None) != (reference_codes is None):
            raise ValueError("reference and reference_codes must be provided together")
        self._dia = dia_model
        self._reference = reference
        self._reference_codes = reference_codes
        self._compiled = compiled
        self.last_stats: StreamStats | None = None

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def audio_context_tokens(self) -> int:
        return int(self._dia.config.data.audio_length)

    @property
    def reference(self) -> VoiceReference | None:
        return self._reference

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
        audio_context_tokens: int | None = DEFAULT_AUDIO_CONTEXT_TOKENS,
        reference_manifest: str | Path | None = DEFAULT_REFERENCE_MANIFEST,
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
        if audio_context_tokens is not None:
            requested = int(audio_context_tokens)
            if requested <= 0:
                raise ValueError("audio_context_tokens must be > 0")
            # Dia's state/cache tensors require a multiple of 128.
            requested = ((requested + 127) // 128) * 128
            data_config = engine.config.data.model_copy(
                update={"audio_length": requested}
            )
            runtime_config = engine.config.model_copy(update={"data": data_config})
            engine.config = runtime_config
            engine.model.config = runtime_config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        engine.device = device
        engine.model = engine.model.to(device).eval()
        engine.dac_model = engine.dac_model.to(device)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        reference = (
            VoiceReference.load(reference_manifest)
            if reference_manifest is not None
            else None
        )
        if reference is not None and reference.language != "sw":
            raise ValueError(
                f"Unsupported reference language {reference.language!r}; expected 'sw'"
            )
        reference_codes = (
            engine.load_audio(str(reference.audio_path))
            if reference is not None
            else None
        )
        wrapper = cls(
            engine,
            reference=reference,
            reference_codes=reference_codes,
            compiled=False,
        )
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
        cfg_filter_top_k: int = 45,
        seed: int | None = None,
        max_tokens: int | None = None,
        segment_max_bytes: int | None = DEFAULT_SEGMENT_MAX_BYTES,
        use_torch_compile: bool | None = None,
        stats: StreamStats | None = None,
    ) -> Iterator[np.ndarray]:
        overall_started = time.perf_counter()
        text_limit = int(self._dia.config.data.text_length)
        clone_prefix = (
            format_sw_prompt(self._reference.transcript)
            if self._reference is not None
            else ""
        )
        # Use raw UTF-8 length as a conservative bound.  The [sw] marker is
        # collapsed to one token later, so this never underestimates capacity.
        available_text_bytes = text_limit - len(clone_prefix.encode("utf-8")) - 8
        if available_text_bytes < 16:
            raise ValueError(
                "reference transcript is too long for Dia's text context"
            )
        split_budget = available_text_bytes
        if segment_max_bytes is not None:
            if int(segment_max_bytes) < 16:
                raise ValueError("segment_max_bytes must be at least 16 or None")
            split_budget = min(split_budget, int(segment_max_bytes))
        segments = split_for_tts(text, split_budget)
        if not segments:
            raise ValueError("text must be non-empty")

        if self._reference is not None:
            prompts = [
                format_clone_prompt(self._reference.transcript, part)
                for part in segments
            ]
        else:
            prompts = [format_sw_prompt(part) for part in segments]
        use_compile = self._compiled if use_torch_compile is None else use_torch_compile
        overall = stats if stats is not None else StreamStats()
        overall.reset(self.sample_rate)
        self.last_stats = overall
        try:
            for prompt in prompts:
                segment_stats = StreamStats()
                for chunk in stream_utterance(
                    self._dia,
                    prompt,
                    chunk_ms=chunk_ms,
                    max_tokens=max_tokens,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=cfg_filter_top_k,
                    seed=seed,
                    audio_prompt=self._reference_codes,
                    use_torch_compile=use_compile,
                    stats=segment_stats,
                ):
                    if overall.chunks_emitted == 0:
                        overall.time_to_first_chunk_s = time.perf_counter() - overall_started
                    overall.samples_emitted += int(chunk.size)
                    overall.chunks_emitted += 1
                    yield chunk
                overall.frames_generated += segment_stats.frames_generated
                overall.prompt_frames += segment_stats.prompt_frames
                overall.segments_generated += 1
                overall.hit_token_limit |= segment_stats.hit_token_limit
                overall.stop_reason = (
                    "token_limit"
                    if overall.hit_token_limit
                    else segment_stats.stop_reason
                )
        finally:
            overall.total_s = time.perf_counter() - overall_started

    def stream_pcm16(self, text: str, **kwargs) -> Iterator[bytes]:
        for chunk in self.stream(text, **kwargs):
            pcm = np.round(np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2")
            yield pcm.tobytes()
