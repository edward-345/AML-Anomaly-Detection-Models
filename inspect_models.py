import json
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

with open("runs/lof_runs.jsonl") as f:
    runs = [json.loads(line) for line in f]

df = pd.json_normalize(runs)
df