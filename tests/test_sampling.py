from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dia.model import Dia, _sample_next_token


class SamplingTests(unittest.TestCase):
    def test_eos_is_not_randomly_sampled(self) -> None:
        logits = torch.tensor(
            [
                [5.0, 4.0, 3.0, 2.0, 4.5],  # EOS plausible, but not highest
                [1.0, 2.0, 3.0, 4.0, 5.0],  # EOS highest
            ]
        )
        seen: list[torch.Tensor] = []

        def choose_argmax(probs: torch.Tensor, num_samples: int) -> torch.Tensor:
            seen.append(probs.detach().clone())
            return torch.argmax(probs, dim=-1, keepdim=True)

        with patch("torch.multinomial", side_effect=choose_argmax):
            result = _sample_next_token(
                logits,
                temperature=1.0,
                top_p=1.0,
                audio_eos_value=4,
            )

        self.assertEqual(result.tolist(), [0, 4])
        self.assertEqual(float(seen[0][0, 4]), 0.0)
        self.assertTrue(torch.all(seen[0][1, :4] == 0))

    def test_cfg_filters_candidates_but_samples_conditional_logits(self) -> None:
        # Guided logits strongly prefer token 0, while the original conditional
        # distribution prefers token 1. Both enter guided top-2; correct CFG
        # filtering therefore returns token 1 at temperature zero.
        uncond = torch.tensor([[[[-10.0, 3.0, 0.0, -1.0, -5.0, -9.0]]]])
        cond = torch.tensor([[[[1.0, 3.0, 0.0, -1.0, -5.0, -9.0]]]])
        logits = torch.cat([uncond, cond], dim=0)

        decoder = SimpleNamespace(decode_step=lambda *args, **kwargs: logits)
        fake_dia = SimpleNamespace(
            config=SimpleNamespace(data=SimpleNamespace(audio_eos_value=4)),
            model=SimpleNamespace(decoder=decoder),
        )
        result = Dia._decoder_step(
            fake_dia,
            torch.zeros((2, 1, 1), dtype=torch.long),
            SimpleNamespace(),
            cfg_scale=1.0,
            temperature=0.0,
            top_p=1.0,
            cfg_filter_top_k=2,
            current_idx=torch.tensor([0]),
        )
        self.assertEqual(result.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
