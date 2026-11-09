"""Entry point: python scripts/aopt.py <command> (see aoptcore/cli.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoptcore.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
