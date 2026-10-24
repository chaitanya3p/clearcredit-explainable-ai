"""Run the synthetic fairness audit without using demographic data for modeling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import _fairness_report  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(_fairness_report(), indent=2))
