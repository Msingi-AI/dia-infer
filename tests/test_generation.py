from __future__ import annotations

import inspect
import unittest

import serve
from dia.generation import DEFAULT_GENERATION_CONFIG, GenerationConfig
from dia.model import Dia
from dia_infer.engine import DiaEngine
from dia_infer.stream import stream_utterance


class GenerationConfigTests(unittest.TestCase):
    def test_defaults_are_shared_by_public_generation_entry_points(self) -> None:
        expected = DEFAULT_GENERATION_CONFIG.temperature
        self.assertEqual(
            inspect.signature(Dia.generate).parameters["temperature"].default,
            expected,
        )
        self.assertEqual(
            inspect.signature(DiaEngine.stream).parameters["temperature"].default,
            expected,
        )
        self.assertEqual(
            inspect.signature(stream_utterance).parameters["temperature"].default,
            expected,
        )
        self.assertEqual(
            serve.GenerateRequest.model_fields["temperature"].default,
            serve.SERVICE_GENERATION_CONFIG.temperature,
        )

    def test_environment_and_request_mapping_override_one_base(self) -> None:
        configured = GenerationConfig.from_env(
            {
                "DIA_TEMPERATURE": "0.9",
                "DIA_SEED": "42",
                "DIA_CHUNK_MS": "500",
            }
        )
        self.assertEqual(configured.temperature, 0.9)
        self.assertEqual(configured.seed, 42)
        self.assertEqual(configured.chunk_ms, 500)
        self.assertEqual(configured.cfg_scale, DEFAULT_GENERATION_CONFIG.cfg_scale)

        request = GenerationConfig.from_mapping(
            {"temperature": 1.0, "segment_max_bytes": None},
            base=configured,
        )
        self.assertEqual(request.temperature, 1.0)
        self.assertIsNone(request.segment_max_bytes)
        self.assertEqual(request.seed, 42)

    def test_invalid_settings_fail_at_configuration_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_p"):
            GenerationConfig(top_p=0)
        with self.assertRaisesRegex(ValueError, "segment_max_bytes"):
            GenerationConfig(segment_max_bytes=8)

    def test_request_model_creates_generation_config(self) -> None:
        request = serve.GenerateRequest(
            text="Habari.",
            temperature=0.8,
            seed=37,
            chunk_ms=500,
        )
        generation = request.generation_config()
        self.assertEqual(generation.temperature, 0.8)
        self.assertEqual(generation.seed, 37)
        self.assertEqual(generation.chunk_ms, 500)


if __name__ == "__main__":
    unittest.main()
