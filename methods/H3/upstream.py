"""Import the local SolarWM source."""
import sys
from .config import path


def activate(root):
    root = path(root).resolve()
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return root
