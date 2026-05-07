"""Re-grade NOT_ATTEMPTED (C) cases that may be grader noise.

For every .simpleqa*.jsonl in results/, find rows with grade=='C' and call
the grader again. If the second call returns A or B, update the row.
Recompute and overwrite the corresponding .summary.json.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load .env
env = ROOT.parent / ".env"
for line in env.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)

from grader import simpleqa_grade


def regrade_file(path: Path, n_parallel: int = 8) -> dict:
    rows = [json.loads(l) for l in path.open()]
    targets = [(i, r) for i, r in enumerate(rows) if r.get("grade") == "C" and r.get("response")]
    if not targets:
        return {"path": str(path), "n_C": 0, "changed": 0}

    changed = 0
    new_grades = {}

    def work(i, r):
        try:
            g = simpleqa_grade(r["question"], r["target"], r["response"])
            return i, g
        except Exception as e:
            return i, "C"

    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        futs = [ex.submit(work, i, r) for i, r in targets]
        for fut in as_completed(futs):
            i, g = fut.result()
            new_grades[i] = g
            if g != "C":
                changed += 1

    # apply updates
    for i, g in new_grades.items():
        rows[i]["grade"] = g

    # rewrite jsonl
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # recompute summary
    n = len(rows)
    a = sum(1 for r in rows if r.get("grade") == "A")
    b = sum(1 for r in rows if r.get("grade") == "B")
    c = sum(1 for r in rows if r.get("grade") == "C")
    summary = {
        "n": n, "correct": a, "incorrect": b, "not_attempted": c,
        "accuracy": a / max(n, 1),
        "f_score": 2 * a / max(2 * a + b + c, 1),
    }
    sum_path = path.with_suffix(".summary.json")
    sum_path.write_text(json.dumps(summary, indent=2))

    return {"path": str(path), "n_C": len(targets), "changed": changed,
            "new_acc": summary["accuracy"], "new_F": summary["f_score"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=str(ROOT.parent / "results"))
    ap.add_argument("--pattern", default="*.simpleqa*.jsonl",
                    help="glob pattern under results_dir")
    ap.add_argument("--n_parallel", type=int, default=8)
    args = ap.parse_args()

    files = sorted(Path(args.results_dir).glob(args.pattern))
    print(f"[regrade] found {len(files)} jsonl files")
    for f in files:
        info = regrade_file(f, args.n_parallel)
        if info["n_C"] > 0:
            print(f"  {f.name}: C={info['n_C']:3d}  changed={info['changed']:3d}  "
                  f"new_acc={info['new_acc']*100:.2f}%  new_F={info['new_F']*100:.2f}")
        else:
            print(f"  {f.name}: no C cases")


if __name__ == "__main__":
    main()
