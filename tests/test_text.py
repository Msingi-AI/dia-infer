from __future__ import annotations

import unittest

from dia_infer.text import estimate_spoken_seconds, normalize_for_tts, split_for_tts


class TextSegmentationTests(unittest.TestCase):
    def test_segments_preserve_all_text_and_fit_budget(self) -> None:
        text = (
            "Hii ni sentensi ndefu sana, yenye maneno mengi ambayo yanapaswa "
            "kugawanywa vizuri bila kupoteza neno lolote. Halafu inaendelea."
        )
        segments = split_for_tts(text, max_bytes=55)

        self.assertGreater(len(segments), 1)
        self.assertEqual(" ".join(segments), normalize_for_tts(text))
        self.assertTrue(all(len(part.encode("utf-8")) <= 55 for part in segments))

    def test_unicode_budget_counts_utf8_bytes(self) -> None:
        text = "Maelezo ya ng'ombe yameandikwa kwa uangalifu sana na yanaendelea."
        segments = split_for_tts(text, max_bytes=32)
        self.assertTrue(all(len(part.encode("utf-8")) <= 32 for part in segments))

    def test_long_text_targets_moderate_audio_duration(self) -> None:
        sentence = "Tunazungumza kwa utulivu ili kila mtu aweze kuelewa vizuri."
        text = " ".join([sentence] * 12)
        segments = split_for_tts(text, max_bytes=512)

        self.assertGreater(len(segments), 1)
        self.assertEqual(" ".join(segments), normalize_for_tts(text))
        self.assertTrue(
            all(estimate_spoken_seconds(part) <= 15.0 for part in segments)
        )

    def test_short_conversational_text_is_not_padded_or_rewritten(self) -> None:
        text = "Niko sawa, asante."
        self.assertEqual(split_for_tts(text, max_bytes=220), [text])

    def test_empty_text_has_no_segments(self) -> None:
        self.assertEqual(split_for_tts("   ", max_bytes=32), [])


if __name__ == "__main__":
    unittest.main()
