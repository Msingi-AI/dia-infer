"""dia-infer: streaming Dia TTS for msingiai/dia (nari@052a840 + LANG2BYTE)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dia_infer.text import format_sw_prompt, normalize_for_tts

if TYPE_CHECKING:
    from dia_infer.engine import DiaEngine

SAMPLE_RATE = 44_100

__all__ = ["DiaEngine", "SAMPLE_RATE", "format_sw_prompt", "normalize_for_tts"]


def __getattr__(name: str):
    if name == "DiaEngine":
        from dia_infer.engine import DiaEngine

        return DiaEngine
    raise AttributeError(name)
