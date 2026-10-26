"""pytest setup: add the repository root to sys.path, so that `import config` and `import pipeline...` work."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
