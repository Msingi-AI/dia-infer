from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dia_infer.stream import _decoder_step_for


class CompiledDecoderTests(unittest.TestCase):
    def test_compiled_wrapper_is_reused_for_same_engine_and_mode(self) -> None:
        decoder_step = object()
        engine = SimpleNamespace(_decoder_step=decoder_step)
        compiled = object()

        with patch("dia_infer.stream.torch.compile", return_value=compiled) as compile_fn:
            first = _decoder_step_for(engine, "max-autotune")
            second = _decoder_step_for(engine, "max-autotune")

        self.assertIs(first, compiled)
        self.assertIs(second, compiled)
        compile_fn.assert_called_once_with(
            decoder_step,
            fullgraph=True,
            mode="max-autotune",
        )

    def test_compile_modes_have_separate_cached_wrappers(self) -> None:
        engine = SimpleNamespace(_decoder_step=object())
        with patch(
            "dia_infer.stream.torch.compile",
            side_effect=["max", "reduced"],
        ) as compile_fn:
            self.assertEqual(_decoder_step_for(engine, "max-autotune"), "max")
            self.assertEqual(_decoder_step_for(engine, "reduce-overhead"), "reduced")

        self.assertEqual(compile_fn.call_count, 2)


if __name__ == "__main__":
    unittest.main()
