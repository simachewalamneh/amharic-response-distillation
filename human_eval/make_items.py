import csv, json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = HERE
PER_TASK = 10
random.seed(11)

def read(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

prompts = {r["eval_id"]: r["prompt"] for r in read(f"{REPO}/eval_prompts_60.csv")}
preds = read(f"{REPO}/results/simachew.csv")

by_task = {}
for r in sorted(preds, key=lambda r: r["eval_id"]):
    by_task.setdefault(r["task"], []).append(r)

picked = []
for task, rows in by_task.items():
    picked += random.sample(rows, min(PER_TASK, len(rows)))
random.shuffle(picked)

items, key = [], []
for r in picked:
    swap = random.random() < 0.5
    first, second = ("distilled", "base") if swap else ("base", "distilled")
    items.append({"eval_id": r["eval_id"], "task": r["task"], "prompt": prompts[r["eval_id"]],
                  "A": r[f"{first}_output"], "B": r[f"{second}_output"]})
    key.append({"eval_id": r["eval_id"], "A_is": first, "B_is": second})

json.dump(items, open(f"{OUT}/rating_items.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(f"{OUT}/rating_key.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["eval_id", "A_is", "B_is"])
    w.writeheader()
    w.writerows(key)
print(len(items), "items written to", OUT)
