"""Run from constraintKB/create_kb/ or from the repository root."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from constraintKB.create_kb.constraint_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
