# Copyright (c) 2026 Franklin Baldo. See LICENSE.
# /// script
# requires-python = ">=3.11"
# dependencies = ["hilda-ablation", "sentence-transformers", "torch"]
#
# [tool.uv.sources]
# hilda-ablation = { path = "..", editable = true }
# ///
"""Run the representation ablation end to end.

Usage:
    uv run scripts/run_ablation.py --corpus-size 8000 --queries 200
"""

from __future__ import annotations

import sys

from hilda_ablation.cli import main

if __name__ == "__main__":
    sys.exit(main())
