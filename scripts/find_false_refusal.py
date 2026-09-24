import json
from pathlib import Path

results = json.loads(Path("evaluation/last_run_results.json").read_text())

for r in results:
    if r.get("incorrectly_refused"):
        print(f"[{r['id']}] {r['question']}")
        print(f"answer_preview: {r['answer_preview']}")