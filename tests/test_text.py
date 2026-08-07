from __future__ import annotations

import unittest

from dia_infer.text import normalize_for_tts, split_for_tts


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

    def test_empty_text_has_no_segments(self) -> None:
        self.assertEqual(split_for_tts("   ", max_bytes=32), [])


if __name__ == "__main__":
    unittest.main()
