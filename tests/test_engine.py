from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from dia.generation import GenerationConfig
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
    def test_generation_config_reaches_streamer(self) -> None:
        engine = DiaEngine(_FakeDia())
        generation = GenerationConfig(
            temperature=0.8,
            cfg_scale=2.5,
            top_p=0.9,
            cfg_filter_top_k=30,
            seed=37,
            max_tokens=400,
            segment_max_bytes=180,
            chunk_ms=500,
        )

        def fake_stream(_dia, _prompt: str, **kwargs):
            stats: StreamStats = kwargs["stats"]
            stats.stop_reason = "eos"
            yield np.ones(32, dtype=np.float32)

        with patch(
            "dia_infer.engine.stream_utterance", side_effect=fake_stream
        ) as stream:
            list(engine.stream("Habari.", generation_config=generation))

        kwargs = stream.call_args.kwargs
        self.assertEqual(kwargs["temperature"], generation.temperature)
        self.assertEqual(kwargs["cfg_scale"], generation.cfg_scale)
        self.assertEqual(kwargs["top_p"], generation.top_p)
        self.assertEqual(kwargs["cfg_filter_top_k"], generation.cfg_filter_top_k)
        self.assertEqual(kwargs["seed"], generation.seed)
        self.assertEqual(kwargs["max_tokens"], generation.max_tokens)
        self.assertEqual(kwargs["chunk_ms"], generation.chunk_ms)

    def test_generation_config_cannot_mix_with_legacy_overrides(self) -> None:
        engine = DiaEngine(_FakeDia())
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            list(
                engine.stream(
                    "Habari.",
                    generation_config=GenerationConfig(),
                    temperature=0.8,
                )
            )

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


class EngineWarmupTests(unittest.TestCase):
    def test_warmup_stops_after_first_chunk_and_uses_requested_options(self) -> None:
        engine = DiaEngine(_FakeDia())
        consumed: list[int] = []
        closed: list[bool] = []

        def chunks():
            try:
                consumed.append(1)
                yield np.ones(8, dtype=np.float32)
                consumed.append(2)
                yield np.ones(8, dtype=np.float32)
            finally:
                closed.append(True)

        options = {
            "temperature": 1.0,
            "cfg_scale": 3.0,
            "top_p": 0.95,
            "cfg_filter_top_k": 45,
        }
        with patch.object(engine, "stream", return_value=chunks()) as stream:
            engine._warm_compile("Habari.", options)

        stream.assert_called_once_with(
            "Habari.",
            use_torch_compile=True,
            **options,
        )
        self.assertEqual(consumed, [1])
        self.assertEqual(closed, [True])
        self.assertTrue(engine._compiled)

    def test_warmup_rejects_internal_stream_overrides(self) -> None:
        engine = DiaEngine(_FakeDia())
        with self.assertRaisesRegex(ValueError, "use_torch_compile"):
            engine._warm_compile(
                "Habari.",
                {"use_torch_compile": False},
            )


if __name__ == "__main__":
    unittest.main()
