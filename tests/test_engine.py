from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from dia_infer.engine import DiaEngine
from dia_infer.reference import VoiceReference
from dia_infer.stream import StreamStats


class _FakeModel:
    def eval(self) -> "_FakeModel":
        return self


class _FakeDia:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            data=SimpleNamespace(text_length=512, audio_length=3072)
        )
        self.model = _FakeModel()
        self.device = torch.device("cpu")


class EngineSegmentationTests(unittest.TestCase):
    def test_long_input_reuses_one_encoded_reference(self) -> None:
        dia = _FakeDia()
        reference_codes = torch.zeros((10, 9), dtype=torch.int32)
        engine = DiaEngine(
            dia,
            reference=VoiceReference(
                id="test-voice",
                language="sw",
                audio_path=Path("reference.wav"),
                transcript="Hii ndiyo sauti ya marejeo.",
            ),
            reference_codes=reference_codes,
        )
        seen_prompts: list[str] = []
        seen_refs: list[torch.Tensor] = []

        def fake_stream(_dia, prompt: str, **kwargs):
            seen_prompts.append(prompt)
            seen_refs.append(kwargs["audio_prompt"])
            stats: StreamStats = kwargs["stats"]
            stats.frames_generated = 20
            stats.prompt_frames = 10
            stats.stop_reason = "eos"
            yield np.ones(32, dtype=np.float32)

        text = " ".join(["maneno"] * 70)
        stats = StreamStats()
        with patch("dia_infer.engine.stream_utterance", side_effect=fake_stream):
            chunks = list(
                engine.stream(
                    text,
                    segment_max_bytes=80,
                    stats=stats,
                )
            )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(ref is reference_codes for ref in seen_refs))
        self.assertTrue(all(prompt.count("[sw]") == 2 for prompt in seen_prompts))
        self.assertEqual(stats.segments_generated, len(chunks))
        self.assertEqual(stats.samples_emitted, 32 * len(chunks))
        self.assertEqual(stats.stop_reason, "eos")


if __name__ == "__main__":
    unittest.main()
