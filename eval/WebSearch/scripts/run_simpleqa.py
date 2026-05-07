"""Run SimpleQA evaluation against an OpenAI-compatible (vLLM) endpoint."""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from agent import answer_with_tools
from grader import simpleqa_grade


def load_dataset(csv_path: Path, limit: int | None = None):
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"problem": row["problem"], "answer": row["answer"]})
    if limit:
        rows = rows[:limit]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="served model name on vLLM (e.g. 'eval_target')")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--dataset", default=str(Path(__file__).resolve().parent.parent / "data" / "simple_qa_test_set.csv"))
    ap.add_argument("--limit", type=int, default=None, help="limit number of questions (for smoke test)")
    ap.add_argument("--n_parallel", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument("--output", required=True, help="output JSONL path for per-question results")
    args = ap.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=args.endpoint)
    rows = load_dataset(Path(args.dataset), args.limit)
    print(f"[simpleqa] {len(rows)} questions  parallel={args.n_parallel}")

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w")

    def work(i, row):
        t0 = time.time()
        out = answer_with_tools(client, args.model, row["problem"],
                                 max_steps=args.max_steps, temperature=args.temperature)
        try:
            grade = simpleqa_grade(row["problem"], row["answer"], out["response"])
        except Exception as e:
            grade = "C"; out.setdefault("trace", []).append({"grader_error": str(e)})
        rec = {"i": i, "question": row["problem"], "target": row["answer"],
               "response": out["response"], "grade": grade,
               "elapsed": round(time.time()-t0, 1), "trace": out.get("trace", [])}
        return rec

    n_a = n_b = n_c = 0
    with ThreadPoolExecutor(max_workers=args.n_parallel) as ex:
        futures = {ex.submit(work, i, r): i for i, r in enumerate(rows)}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"i": futures[fut], "error": f"{type(e).__name__}: {e}", "grade": "C"}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            g = rec.get("grade", "C")
            if   g == "A": n_a += 1
            elif g == "B": n_b += 1
            else:          n_c += 1
            if k % 20 == 0 or k == len(rows):
                tot = n_a + n_b + n_c
                print(f"[simpleqa] {k}/{len(rows)}  CORRECT={n_a/tot*100:.1f}%  INCORRECT={n_b/tot*100:.1f}%  NOT_ATTEMPTED={n_c/tot*100:.1f}%")

    fout.close()
    summary = {"n": n_a+n_b+n_c, "correct": n_a, "incorrect": n_b, "not_attempted": n_c,
                "accuracy": n_a / max(n_a+n_b+n_c, 1),
                "f_score": 2*n_a / max(2*n_a + n_b + n_c, 1)}
    sum_path = out_path.with_suffix(".summary.json")
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"[simpleqa] DONE  accuracy={summary['accuracy']*100:.2f}%  F={summary['f_score']*100:.2f}  → {sum_path}")


if __name__ == "__main__":
    main()
