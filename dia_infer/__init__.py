"""dia-infer: streaming Dia TTS engine for msingiai/dia (nari@052a840 + LANG2BYTE).

Local laptop GPUs (e.g. RTX 5060 8 GB) do **not** meet realtime for this model.
Production path: Modal A10 WebSocket (`modal_app.py`). Local `infer.py` is smoke only.
"""

from dia_infer.engine import DiaEngine, SAMPLE_RATE

__all__ = ["DiaEngine", "SAMPLE_RATE"]
