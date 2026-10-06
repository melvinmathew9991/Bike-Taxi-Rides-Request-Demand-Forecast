#!/usr/bin/env python
"""
Kept so `python run_pipeline.py ...` still works. Prefer `biketaxi run ...`.

The command line lives in `ML_Pipeline.cli.run`. Like every other entry point
it needs the package installed (`pip install -e .`); it no longer puts `src/`
on `sys.path` by hand.
"""

import sys

from ML_Pipeline.cli.run import main

if __name__ == "__main__":
    sys.exit(main())
