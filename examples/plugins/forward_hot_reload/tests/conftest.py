import sys
from pathlib import Path

# The package next to tests/, importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
