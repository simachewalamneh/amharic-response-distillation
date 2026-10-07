import csv, json, os, statistics as st

OUT = os.path.dirname(os.path.abspath(__file__))
def read(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

key = {r["eval_id"]: r for r in read(f"{OUT}/rating_key.csv")}
task_of = {r["eval_id"]: r["task"] for r in json.load(open(f"{OUT}/rating_items.json", encoding="utf-8"))}
ratings = read(f"{OUT}/ratings.csv")

vals, wins = {}, {}
for r in ratings:
    k, task = key[r["eval_id"]], task_of[r["eval_id"]]
    sc = {}
    for side in "AB":
        model = k[f"{side}_is"]
        for m in ("fluency", "correct"):
            v = int(r[f"{m}_{side}"])
            sc[(model, m)] = v
            for t in (task, "OVERALL"):
                vals.setdefault((t, model, m), []).append(v)
    d, b = sc[("distilled", "correct")], sc[("base", "correct")]
    for t in (task, "OVERALL"):
        wins.setdefault(t, []).append("distilled" if d > b else "base" if b > d else "tie")

out_rows = []
print(f"{'task':24s} {'n':>3s} | {'fluency base':>12s} {'distilled':>9s} | {'correct base':>12s} {'distilled':>9s} | wins d/tie/b")
for t in sorted({t for t, _, _ in vals} - {"OVERALL"}) + ["OVERALL"]:
    n = len(wins[t])
    fb, fd = st.mean(vals[(t, "base", "fluency")]), st.mean(vals[(t, "distilled", "fluency")])
    cb, cd = st.mean(vals[(t, "base", "correct")]), st.mean(vals[(t, "distilled", "correct")])
    w = wins[t]
    print(f"{t:24s} {n:3d} | {fb:12.2f} {fd:9.2f} | {cb:12.2f} {cd:9.2f} | {w.count('distilled')}/{w.count('tie')}/{w.count('base')}")
    out_rows.append({"task": t, "n": n, "fluency_base": round(fb, 2), "fluency_distilled": round(fd, 2),
                     "correct_base": round(cb, 2), "correct_distilled": round(cd, 2),
                     "wins_distilled": w.count("distilled"), "ties": w.count("tie"), "wins_base": w.count("base")})
with open(f"{OUT}/rating_summary.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
    w.writeheader()
    w.writerows(out_rows)
print("\nsaved", f"{OUT}/rating_summary.csv")
