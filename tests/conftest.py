from __future__ import annotations

import sys
from pathlib import Path

# Project root on sys.path so `import onnx_cut` works without an install.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
