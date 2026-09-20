"""Makes the src package importable from notebooks and tests without installation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
