import json
from pathlib import Path

results = json.loads(Path("evaluation/last_run_results.json").read_text())
for r in results:
    if r["id"] == "q26":
        print(f"Question: {r['question']}")
        print(f"Full answer preview: {r['answer_preview']}")