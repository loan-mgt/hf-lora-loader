"""Expose node mappings for ComfyUI while supporting both pip and local installs."""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

__author__ = """Loan Maeght"""
__email__ = "qypol342@gmail.com"
__version__ = "0.0.1"

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists():  # allow running directly from a ComfyUI custom_nodes checkout
    sys.path.insert(0, str(_SRC))

from hf_lora_loader.nodes import NODE_CLASS_MAPPINGS
from hf_lora_loader.nodes import NODE_DISPLAY_NAME_MAPPINGS


