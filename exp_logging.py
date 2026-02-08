import json
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

LOG_FILE = RUNS_DIR / "lof_runs.jsonl"


def log_run(payload: dict):
    """
    Appends one experiment run to a JSONL file.
    """
    payload = payload.copy()
    payload["timestamp"] = datetime.now().isoformat()

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(payload) + "\n")