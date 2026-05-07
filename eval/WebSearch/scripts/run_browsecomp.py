"""Run BrowseComp evaluation. Dataset is XOR-encrypted with `canary` field."""
from __future__ import annotations
import argparse, base64, csv, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from agent import answer_with_tools
from grader import browsecomp_grade


def _derive_key(password: str, length: int) -> bytes:
    h = hashlib.sha256(); h.update(password.encode()); k = h.digest()
    return k * (length // len(k)) + k[: length % len(k)]


def _decrypt(b64: str, password: str) -> str:
    enc = base64.b64decode(b64); key = _derive_key(password, len(enc))
    return bytes(a ^ b for a, b in zip(enc, key)).decode()


QUERY_TEMPLATE = """\
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
"""


def load_dataset(csv_path: Path, limit: int | None = None):
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                problem = _decrypt(row["problem"], row["canary"])
                answer  = _decrypt(row["answer"],  row["canary"])
            except Exception as e:
                continue
            rows.append({"problem": problem, "answer": answer, "topic": row.get("problem_topic", "")})
    if limit:
        rows = rows[:limit]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--dataset", default=str(Path(__file__).resolve().parent.parent / "data" / "browse_comp_test_set.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n_parallel", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=args.endpoint)
    rows = load_dataset(Path(args.dataset), args.limit)
    print(f"[browsecomp] {len(rows)} questions  parallel={args.n_parallel}")

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w")

    def work(i, row):
        t0 = time.time()
        prompt = QUERY_TEMPLATE.format(Question=row["problem"])
        out = answer_with_tools(client, args.model, prompt,
                                 max_steps=args.max_steps, temperature=args.temperature)
        try:
            grade = browsecomp_grade(row["problem"], row["answer"], out["response"])
        except Exception as e:
            grade = "no"; out.setdefault("trace", []).append({"grader_error": str(e)})
        return {"i": i, "topic": row.get("topic", ""), "question": row["problem"],
                "target": row["answer"], "response": out["response"], "grade": grade,
                "elapsed": round(time.time()-t0, 1), "trace": out.get("trace", [])}

    correct = 0
    with ThreadPoolExecutor(max_workers=args.n_parallel) as ex:
        futs = {ex.submit(work, i, r): i for i, r in enumerate(rows)}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"i": futs[fut], "error": f"{type(e).__name__}: {e}", "grade": "no"}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            if rec.get("grade") == "yes":
                correct += 1
            if k % 20 == 0 or k == len(rows):
                print(f"[browsecomp] {k}/{len(rows)}  acc={correct/k*100:.2f}%")

    fout.close()
    summary = {"n": len(rows), "correct": correct, "accuracy": correct / max(len(rows), 1)}
    sum_path = out_path.with_suffix(".summary.json")
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"[browsecomp] DONE  accuracy={summary['accuracy']*100:.2f}%  → {sum_path}")


if __name__ == "__main__":
    main()
