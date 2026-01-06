"""
GXP Consumer (Thin Wrapper)

Thin wrapper entrypoint that routes to regional_electricity_poc implementation.
Allows `python -m src.gxp_consumer` to work as an alias.
"""

from __future__ import annotations

import sys
from pathlib import Path as PathlibPath

ROOT = PathlibPath(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import and call the main function from regional_electricity_poc
from src.regional_electricity_poc import main

if __name__ == '__main__':
    main()

