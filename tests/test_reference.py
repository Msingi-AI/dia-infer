from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dia_infer.reference import VoiceReference


class ReferenceManifestTests(unittest.TestCase):
    def test_relative_audio_path_is_resolved_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "voice.wav").write_bytes(b"RIFF")
            manifest = root / "voice.json"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "voice",
                        "language": "sw",
                        "audio": "voice.wav",
                        "transcript": "Habari.",
                    }
                ),
                encoding="utf-8",
            )

            reference = VoiceReference.load(manifest)

        self.assertEqual(reference.id, "voice")
        self.assertEqual(reference.audio_path, (root / "voice.wav").resolve())

    def test_missing_required_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "voice.json"
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                VoiceReference.load(manifest)

    def test_catalog_selects_voice_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "t6.wav").write_bytes(b"RIFF")
            manifest = root / "voices.json"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "t6",
                        "language": "sw",
                        "voices": {
                            "t6": {
                                "audio": "t6.wav",
                                "transcript": "Habari.",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            reference = VoiceReference.load(manifest)

        self.assertEqual(reference.id, "t6")
        self.assertEqual(reference.audio_path, (root / "t6.wav").resolve())


if __name__ == "__main__":
    unittest.main()
