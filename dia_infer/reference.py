"""Voice-reference manifest loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceReference:
    """An audio prompt and its exact transcript."""

    id: str
    language: str
    audio_path: Path
    transcript: str

    @classmethod
    def load(cls, manifest_path: str | Path) -> "VoiceReference":
        manifest = Path(manifest_path).resolve()
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Reference manifest not found: {manifest}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid reference manifest JSON: {manifest}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Reference manifest must contain a JSON object: {manifest}")

        voice_id = str(data.get("id", "")).strip()
        if "voices" in data:
            voices = data["voices"]
            voice = voices.get(voice_id) if isinstance(voices, dict) else None
            if not isinstance(voice, dict):
                raise ValueError(f"Unknown reference id {voice_id!r}: {manifest}")
            data = {**voice, "id": voice_id, "language": data.get("language")}

        required = ("id", "language", "audio", "transcript")
        missing = [key for key in required if not str(data.get(key, "")).strip()]
        if missing:
            raise ValueError(
                f"Reference manifest {manifest} is missing: {', '.join(missing)}"
            )

        audio_path = (manifest.parent / str(data["audio"])).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        return cls(
            id=str(data["id"]).strip(),
            language=str(data["language"]).strip(),
            audio_path=audio_path,
            transcript=str(data["transcript"]).strip(),
        )
