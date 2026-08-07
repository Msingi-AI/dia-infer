from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest.mock import patch

import serve
from dia_infer.stream import DEFAULT_COMPILE_MODE


class _FakeWebSocket:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.query_params: dict[str, str] = {}

    async def accept(self) -> None:
        self.events.append("accepted")

    async def receive_text(self) -> str:
        return json.dumps({"text": "Habari."})

    async def send_bytes(self, data: bytes) -> None:
        self.events.append(f"sent:{data.decode()}")

    async def send_json(self, data: dict) -> None:
        self.events.append(f"json:{data['event']}")

    async def close(self) -> None:
        self.events.append("closed")


class _FakeEngine:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def stream_pcm16(self, text: str, **kwargs):
        self.events.append("generated:a")
        yield b"a"
        self.events.append("generated:b")
        yield b"b"


class StreamingEndpointTests(unittest.TestCase):
    def tearDown(self) -> None:
        serve._engine = None

    def test_engine_warmup_uses_service_generation_defaults(self) -> None:
        fake_engine = object()
        with patch.object(
            serve.DiaEngine,
            "load",
            return_value=fake_engine,
        ) as load:
            self.assertIs(serve.get_engine(), fake_engine)

        self.assertEqual(
            load.call_args.kwargs["warmup_options"],
            serve._generation_defaults(),
        )
        self.assertEqual(
            load.call_args.kwargs["compile_mode"],
            os.environ.get("DIA_COMPILE_MODE", DEFAULT_COMPILE_MODE),
        )

    def test_chunks_are_sent_before_generation_finishes(self) -> None:
        events: list[str] = []
        ws = _FakeWebSocket(events)
        with patch("serve.get_engine", return_value=_FakeEngine(events)):
            asyncio.run(serve.tts_ws(ws))

        self.assertLess(events.index("sent:a"), events.index("generated:b"))
        self.assertLess(events.index("sent:b"), events.index("json:end"))


if __name__ == "__main__":
    unittest.main()
