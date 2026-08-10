"""DiaEngine — load msingiai/dia and stream audio chunks."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dia.generation import DEFAULT_GENERATION_CONFIG, GenerationConfig
from dia_infer.reference import VoiceReference
from dia_infer.stream import DEFAULT_COMPILE_MODE, StreamStats, stream_utterance
from dia_infer.text import format_clone_prompt, format_sw_prompt, split_for_tts

SAMPLE_RATE = 44_100
DEFAULT_AUDIO_CONTEXT_TOKENS = 3_072
DEFAULT_SEGMENT_MAX_BYTES = DEFAULT_GENERATION_CONFIG.segment_max_bytes
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
        compile_mode: str = DEFAULT_COMPILE_MODE,
    ) -> None:
        if (reference is None) != (reference_codes is None):
            raise ValueError("reference and reference_codes must be provided together")
        self._dia = dia_model
        self._reference = reference
        self._reference_codes = reference_codes
        self._compiled = compiled
        self._compile_mode = compile_mode
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
    def compile_mode(self) -> str:
        return self._compile_mode

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
        compile_mode: str = DEFAULT_COMPILE_MODE,
        warmup_text: str = "Habari.",
        warmup_options: Mapping[str, Any] | None = None,
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
            compile_mode=compile_mode,
        )
        if compile and device.type == "cuda":
            wrapper._warm_compile(warmup_text, warmup_options)
        return wrapper

    def _warm_compile(
        self,
        text: str,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        """Compile production settings and stop after the first audio chunk."""
        warmup_options = dict(options or {})
        reserved = {"stats", "use_torch_compile"}.intersection(warmup_options)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"warmup_options cannot override: {names}")
        chunks = self.stream(
            text,
            use_torch_compile=True,
            **warmup_options,
        )
        try:
            next(chunks)
        except StopIteration as exc:
            raise RuntimeError("Dia warmup produced no audio") from exc
        finally:
            chunks.close()
        self._compiled = True

    def stream(
        self,
        text: str,
        *,
        generation_config: GenerationConfig | None = None,
        chunk_ms: int = DEFAULT_GENERATION_CONFIG.chunk_ms,
        cfg_scale: float = DEFAULT_GENERATION_CONFIG.cfg_scale,
        temperature: float = DEFAULT_GENERATION_CONFIG.temperature,
        top_p: float = DEFAULT_GENERATION_CONFIG.top_p,
        cfg_filter_top_k: int = DEFAULT_GENERATION_CONFIG.cfg_filter_top_k,
        seed: int | None = DEFAULT_GENERATION_CONFIG.seed,
        max_tokens: int | None = DEFAULT_GENERATION_CONFIG.max_tokens,
        segment_max_bytes: int | None = DEFAULT_SEGMENT_MAX_BYTES,
        use_torch_compile: bool | None = None,
        stats: StreamStats | None = None,
    ) -> Iterator[np.ndarray]:
        overall_started = time.perf_counter()
        legacy_config = GenerationConfig(
            temperature=temperature,
            cfg_scale=cfg_scale,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            seed=seed,
            max_tokens=max_tokens,
            segment_max_bytes=segment_max_bytes,
            chunk_ms=chunk_ms,
        )
        if generation_config is not None and legacy_config != DEFAULT_GENERATION_CONFIG:
            raise ValueError(
                "generation_config cannot be combined with generation keyword overrides"
            )
        generation = generation_config or legacy_config
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
        if generation.segment_max_bytes is not None:
            split_budget = min(split_budget, generation.segment_max_bytes)
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
                    chunk_ms=generation.chunk_ms,
                    max_tokens=generation.max_tokens,
                    cfg_scale=generation.cfg_scale,
                    temperature=generation.temperature,
                    top_p=generation.top_p,
                    cfg_filter_top_k=generation.cfg_filter_top_k,
                    seed=generation.seed,
                    audio_prompt=self._reference_codes,
                    use_torch_compile=use_compile,
                    compile_mode=self._compile_mode,
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
