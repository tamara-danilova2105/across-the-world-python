import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import json
from main import app


def main_export() -> None:
    with open("openapi.json", "w", encoding="utf-8") as f:
        json.dump(app.openapi(), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main_export()
