"""Shared generation settings for Dia inference entry points."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Mapping


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Validated sampling, streaming, and segmentation settings."""

    temperature: float = 1.3
    cfg_scale: float = 3.0
    top_p: float = 0.95
    cfg_filter_top_k: int = 45
    seed: int | None = None
    max_tokens: int | None = None
    segment_max_bytes: int | None = 220
    chunk_ms: int = 250

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.cfg_scale < 0:
            raise ValueError("cfg_scale must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.cfg_filter_top_k <= 0:
            raise ValueError("cfg_filter_top_k must be > 0")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0 or None")
        if self.segment_max_bytes is not None and self.segment_max_bytes < 16:
            raise ValueError("segment_max_bytes must be at least 16 or None")
        if self.chunk_ms <= 0:
            raise ValueError("chunk_ms must be > 0")

    def with_overrides(self, **changes: object) -> "GenerationConfig":
        return replace(self, **changes)

    def as_stream_kwargs(self) -> dict[str, object]:
        return asdict(self)

    def as_warmup_options(self) -> dict[str, object]:
        options = self.as_stream_kwargs()
        # The loop bound is not part of the compiled decoder graph. Keeping the
        # warmup bounded by first-chunk consumption is safer than applying a
        # request-specific maximum here.
        options.pop("max_tokens")
        return options

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        *,
        base: "GenerationConfig | None" = None,
    ) -> "GenerationConfig":
        current = base or DEFAULT_GENERATION_CONFIG
        converters = {
            "temperature": float,
            "cfg_scale": float,
            "top_p": float,
            "cfg_filter_top_k": int,
            "seed": int,
            "max_tokens": int,
            "segment_max_bytes": int,
            "chunk_ms": int,
        }
        changes: dict[str, object] = {}
        for name, convert in converters.items():
            if name not in values:
                continue
            value = values[name]
            changes[name] = None if value is None else convert(value)
        return current.with_overrides(**changes)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        base: "GenerationConfig | None" = None,
    ) -> "GenerationConfig":
        source = os.environ if env is None else env
        names = {
            "temperature": "DIA_TEMPERATURE",
            "cfg_scale": "DIA_CFG_SCALE",
            "top_p": "DIA_TOP_P",
            "cfg_filter_top_k": "DIA_CFG_FILTER_TOP_K",
            "seed": "DIA_SEED",
            "max_tokens": "DIA_MAX_TOKENS",
            "segment_max_bytes": "DIA_SEGMENT_MAX_BYTES",
            "chunk_ms": "DIA_CHUNK_MS",
        }
        values = {
            field: source[variable]
            for field, variable in names.items()
            if variable in source and source[variable] != ""
        }
        return cls.from_mapping(values, base=base)


DEFAULT_GENERATION_CONFIG = GenerationConfig()
