#!/usr/bin/env python3
"""
research_figures/generate_figures.py
====================================
Standalone runner to regenerate all verified IEEE publication figures,
predictions CSV, and metrics summary.
"""

import sys
import os
from pathlib import Path

# Auto-switch to project virtual environment if executed with system python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    import subprocess
    sys.exit(subprocess.call([str(VENV_PYTHON), str(PROJECT_ROOT / "scripts" / "generate_all_verified_figures.py")] + sys.argv[1:]))

# If already in virtualenv, invoke main directly
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.generate_all_verified_figures import main

if __name__ == "__main__":
    main()
