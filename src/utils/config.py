from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

