"""Incremental DAC streaming for nari@052a840 stateful Dia API."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch

DEFAULT_CHUNK_MS = 250
DEFAULT_CONTEXT_FRAMES = 16
DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_HOP_LENGTH = 512
DEFAULT_COMPILE_MODE = "max-autotune"


@dataclass
class StreamStats:
    frames_generated: int = 0
    prompt_frames: int = 0
    samples_emitted: int = 0
    chunks_emitted: int = 0
    segments_generated: int = 0
    hit_token_limit: bool = False
    stop_reason: str = ""
    time_to_first_chunk_s: float = 0.0
    total_s: float = 0.0
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @property
    def audio_seconds(self) -> float:
        return self.samples_emitted / max(1, self.sample_rate)

    @property
    def wall_per_audio(self) -> float:
        return self.total_s / max(1e-9, self.audio_seconds)

    def reset(self, sample_rate: int) -> None:
        self.frames_generated = 0
        self.prompt_frames = 0
        self.samples_emitted = 0
        self.chunks_emitted = 0
        self.segments_generated = 0
        self.hit_token_limit = False
        self.stop_reason = ""
        self.time_to_first_chunk_s = 0.0
        self.total_s = 0.0
        self.sample_rate = sample_rate

    def summary(self) -> str:
        return (
            f"steps={self.frames_generated} prompt={self.prompt_frames} "
            f"segments={self.segments_generated} audio={self.audio_seconds:.2f}s "
            f"ttfa={self.time_to_first_chunk_s * 1000:.0f}ms "
            f"total={self.total_s:.2f}s wall/audio={self.wall_per_audio:.2f} "
            f"chunks={self.chunks_emitted} stop={self.stop_reason or 'unknown'}"
        )


def _sample_rate_and_hop(engine: Any) -> tuple[int, int]:
    dac = getattr(engine, "dac_model", None)
    if dac is None:
        raise RuntimeError("Dia must be loaded with DAC enabled")
    return int(getattr(dac, "sample_rate", DEFAULT_SAMPLE_RATE)), int(
        getattr(dac, "hop_length", DEFAULT_HOP_LENGTH)
    )


def _revert_delay(raw_TxC: torch.Tensor, delay: list[int], frame_count: int) -> torch.Tensor:
    columns = [raw_TxC[d : d + frame_count, c] for c, d in enumerate(delay)]
    return torch.stack(columns, dim=1)


def _decode_codes(engine: Any, codes_TxC: torch.Tensor) -> np.ndarray:
    valid = (codes_TxC >= 0) & (codes_TxC <= 1023)
    codes_TxC = torch.where(valid, codes_TxC, torch.zeros_like(codes_TxC))
    with torch.inference_mode():
        if hasattr(engine, "_decode"):
            audio = engine._decode(codes_TxC)
        else:
            codes_BxCxT = codes_TxC.transpose(0, 1).unsqueeze(0)
            z = engine.dac_model.quantizer.from_codes(codes_BxCxT)[0]
            audio = engine.dac_model.decode(z).squeeze()
    return np.ascontiguousarray(audio.detach().float().cpu().numpy().reshape(-1), dtype=np.float32)


class _IncrementalDAC:
    def __init__(
        self,
        engine: Any,
        delay: list[int],
        chunk_frames: int,
        left_context_frames: int,
        right_context_frames: int,
        hop_length: int,
    ) -> None:
        self.engine = engine
        self.delay = delay
        self.max_delay = max(delay, default=0)
        self.chunk_frames = chunk_frames
        self.left_context_frames = left_context_frames
        self.right_context_frames = right_context_frames
        self.hop_length = hop_length
        self.emitted_frames = 0

    def pull(
        self,
        raw_TxC: torch.Tensor,
        *,
        valid_frame_limit: int | None = None,
        final: bool = False,
    ) -> np.ndarray | None:
        complete_frames = max(0, int(raw_TxC.shape[0]) - self.max_delay)
        if valid_frame_limit is not None:
            complete_frames = min(complete_frames, max(0, valid_frame_limit))
        stable = complete_frames if final else max(0, complete_frames - self.right_context_frames)
        pending = stable - self.emitted_frames
        if pending <= 0 or (not final and pending < self.chunk_frames):
            return None
        decode_start = max(0, self.emitted_frames - self.left_context_frames)
        reverted = _revert_delay(raw_TxC, self.delay, complete_frames)
        decoded = _decode_codes(self.engine, reverted[decode_start:complete_frames])
        sample_start = (self.emitted_frames - decode_start) * self.hop_length
        sample_stop = (stable - decode_start) * self.hop_length
        chunk = np.ascontiguousarray(decoded[sample_start:sample_stop], dtype=np.float32)
        self.emitted_frames = stable
        return chunk if chunk.size else None


def _record_chunk(stats: StreamStats, chunk: np.ndarray, started_at: float) -> np.ndarray:
    if stats.chunks_emitted == 0:
        stats.time_to_first_chunk_s = time.perf_counter() - started_at
    stats.chunks_emitted += 1
    stats.samples_emitted += int(chunk.size)
    return chunk


def _maybe_mark_cuda_graph_step() -> None:
    mark = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if mark is not None:
        mark()


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _decoder_step_for(engine: Any, compile_mode: str):
    """Return one persistent compiled decoder wrapper per engine and mode."""
    cache = getattr(engine, "_dia_infer_compiled_steps", None)
    if cache is None:
        cache = {}
        setattr(engine, "_dia_infer_compiled_steps", cache)
    if compile_mode not in cache:
        cache[compile_mode] = torch.compile(
            engine._decoder_step,
            fullgraph=True,
            mode=compile_mode,
        )
    return cache[compile_mode]


def _stream_stateful(
    engine: Any,
    text: str,
    *,
    max_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    use_torch_compile: bool,
    compile_mode: str,
    decoder: _IncrementalDAC,
    stats: StreamStats,
    started_at: float,
    audio_prompt: Any = None,
) -> Iterator[np.ndarray]:
    cfg = engine.config
    delay = list(cfg.data.delay_pattern)
    max_delay = max(delay)
    eos_value = int(cfg.data.audio_eos_value)
    pad_value = int(cfg.data.audio_pad_value)

    prepare_signature = inspect.signature(engine._prepare_generation)
    if "verbose" in prepare_signature.parameters:
        dec_state, dec_output = engine._prepare_generation(text, audio_prompt, False)
    else:
        dec_state, dec_output = engine._prepare_generation(text, audio_prompt)

    prefill_step = int(dec_output.prefill_step)
    if prefill_step + max_delay >= max_tokens:
        raise ValueError(
            "audio prompt leaves no generation budget: "
            f"prompt={prefill_step - 1} frames, context={max_tokens} frames"
        )
    stats.prompt_frames = max(0, prefill_step - 1)
    dec_step = prefill_step - 1
    bos_countdown = max_delay
    eos_detected = False
    eos_countdown = -1
    valid_frame_limit: int | None = None

    needs_current_idx = "current_idx" in inspect.signature(engine._decoder_step).parameters
    current_idx = (
        torch.tensor([dec_step], device=engine.device) if needs_current_idx else None
    )

    step_fn = engine._decoder_step
    if use_torch_compile:
        step_fn = _decoder_step_for(engine, compile_mode)

    while dec_step < max_tokens:
        if eos_countdown == 0:
            break
        current_step = dec_step + 1
        _maybe_mark_cuda_graph_step()
        dec_state.prepare_step(dec_step)
        tokens_Bx1xC = dec_output.get_tokens_at(dec_step).unsqueeze(0).expand(2, -1, -1)
        if needs_current_idx:
            pred_C = step_fn(
                tokens_Bx1xC,
                dec_state,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                current_idx,
            )
            current_idx += 1
        else:
            pred_C = step_fn(
                tokens_Bx1xC, dec_state, cfg_scale, temperature, top_p, cfg_filter_top_k
            )

        reached_limit = current_step >= max_tokens - max_delay
        if (not eos_detected and int(pred_C[0].item()) == eos_value) or reached_limit:
            if not eos_detected:
                eos_detected = True
                eos_countdown = max_delay
                valid_frame_limit = max(0, current_step - prefill_step)
                if reached_limit:
                    stats.hit_token_limit = True
                    stats.stop_reason = "token_limit"
                else:
                    stats.stop_reason = "eos"

        if eos_countdown > 0:
            pred_C = pred_C.clone()
            step_after_eos = max_delay - eos_countdown
            for channel, channel_delay in enumerate(delay):
                if step_after_eos == channel_delay:
                    pred_C[channel] = eos_value
                elif step_after_eos > channel_delay:
                    pred_C[channel] = pad_value
            eos_countdown -= 1

        bos_countdown = max(0, bos_countdown - 1)
        dec_output.update_one(pred_C, current_step, bos_countdown > 0)
        dec_step += 1

        raw = dec_output.generated_tokens[prefill_step : current_step + 1, :]
        stats.frames_generated = int(raw.shape[0])
        chunk = decoder.pull(raw, valid_frame_limit=valid_frame_limit)
        if chunk is not None:
            yield _record_chunk(stats, chunk, started_at)

    raw = dec_output.generated_tokens[prefill_step : dec_step + 1, :]
    chunk = decoder.pull(raw, valid_frame_limit=valid_frame_limit, final=True)
    if chunk is not None:
        yield _record_chunk(stats, chunk, started_at)
    if not stats.stop_reason:
        stats.hit_token_limit = True
        stats.stop_reason = "token_limit"


@torch.inference_mode()
def stream_utterance(
    engine: Any,
    text: str,
    *,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    context_frames: int = DEFAULT_CONTEXT_FRAMES,
    max_tokens: int | None = None,
    cfg_scale: float = 3.0,
    temperature: float = 1.3,
    top_p: float = 0.95,
    cfg_filter_top_k: int = 45,
    seed: int | None = None,
    audio_prompt: Any = None,
    use_torch_compile: bool = False,
    compile_mode: str = DEFAULT_COMPILE_MODE,
    stats: StreamStats | None = None,
) -> Iterator[np.ndarray]:
    """Yield mono float32 waveform chunks @ engine DAC sample rate."""
    if not text.strip():
        raise ValueError("text must be non-empty")
    max_tokens = int(engine.config.data.audio_length) if max_tokens is None else int(max_tokens)
    if max_tokens > int(engine.config.data.audio_length):
        raise ValueError(
            f"max_tokens={max_tokens} exceeds the allocated audio context "
            f"({engine.config.data.audio_length})"
        )
    if cfg_filter_top_k <= 0:
        raise ValueError("cfg_filter_top_k must be > 0")
    sample_rate, hop_length = _sample_rate_and_hop(engine)
    chunk_frames = max(1, round((chunk_ms / 1000.0) * sample_rate / hop_length))
    stream_stats = stats if stats is not None else StreamStats()
    stream_stats.reset(sample_rate)
    started_at = time.perf_counter()
    engine.model.eval()
    _apply_seed(seed)
    delay = list(engine.config.data.delay_pattern)
    decoder = _IncrementalDAC(
        engine, delay, chunk_frames, context_frames, context_frames, hop_length
    )
    try:
        yield from _stream_stateful(
            engine,
            text,
            max_tokens=max_tokens,
            cfg_scale=float(cfg_scale),
            temperature=float(temperature),
            top_p=float(top_p),
            cfg_filter_top_k=int(cfg_filter_top_k),
            use_torch_compile=use_torch_compile,
            compile_mode=compile_mode,
            decoder=decoder,
            stats=stream_stats,
            started_at=started_at,
            audio_prompt=audio_prompt,
        )
    finally:
        stream_stats.total_s = time.perf_counter() - started_at
