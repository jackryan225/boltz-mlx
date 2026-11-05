"""`python -m boltz_mlx.gui` (same as `boltz-mlx gui`, without importing the model code)."""

import sys

from boltz_mlx.gui.main import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
