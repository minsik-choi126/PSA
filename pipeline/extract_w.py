"""W_col extraction with abs + soft weighting (data-parallel across N GPUs).

For each token t in the saved per-expert rollout:

    w[t] = |Δlogp[t]| ^ alpha    where Δlogp[t] = expert_lp[t] - base_lp[t]

α=1.0 (default) is linear in |Δlogp|; α>1 sharpens, α<1 spreads. Every token
contributes proportional to its own |Δlogp| — no top-K threshold, no sign
filtering.

Per layer ℓ and input channel c:

    W_col[i, ℓ, c] = ( Σ_t w[t] · |x_i,ℓ(t)[c]| ) / Σ_t w[t]

Captured via a fused forward hook on every nn.Linear input (no per-call clone).

Per-task data is read from <data_dir>/<task>.npz with this schema (produced
by gen_per_query.py):

    full_tokens     (sum full_seq_lens,) int32
    prompt_lens     (N,)                 int32
    full_seq_lens   (N,)                 int32
    seq_lens        (N,)                 int32      # answer length per query
    expert_names    (E,)                 str
    base_lp         (T,)                 float32    # log p_base on answer tokens
    expert_lp       (E, T)               float32    # log p_E    on answer tokens

Variant-agnostic: pass --experts name=path ... to specify the 3 experts and
the task name corresponding to <task>.npz.
"""
from __future__ import annotations
import argparse, gc, os, pickle, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from helpers import resolve_path, setup_cache  # type: ignore


def parse_experts(experts_args: list) -> list[tuple[str, str]]:
    """Parse --experts NAME=PATH NAME=PATH NAME=PATH → [(name, path), ...]"""
    out = []
    for s in experts_args:
        if "=" not in s:
            raise ValueError(f"--experts entry must be 'name=path'; got {s!r}")
        name, path = s.split("=", 1)
        out.append((name.strip(), path.strip()))
    if len(out) != 3:
        raise ValueError(f"need exactly 3 experts (got {len(out)})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
def load_task(task: str, expert_name: str, alpha: float, data_dir: Path):
    z = np.load(data_dir / f"{task}.npz", allow_pickle=True)
    full_tokens = z["full_tokens"]
    full_seq_lens = [int(x) for x in z["full_seq_lens"]]
    prompt_lens = [int(x) for x in z["prompt_lens"]]
    ans_lens = [int(x) for x in z["seq_lens"]]
    en = [str(n) for n in z["expert_names"]]
    base_lp = z["base_lp"].astype(np.float32)
    exp_lp = z["expert_lp"].astype(np.float32)[en.index(expert_name)]

    delta = exp_lp - base_lp
    w_flat = np.abs(delta).astype(np.float32)
    if alpha != 1.0:
        w_flat = np.power(w_flat, alpha)

    seqs, pls, masks = [], [], []
    off_full, off_ans = 0, 0
    for pl, flen, alen in zip(prompt_lens, full_seq_lens, ans_lens):
        flen, alen = int(flen), int(alen)
        seq = torch.from_numpy(full_tokens[off_full: off_full + flen].astype(np.int64))
        m = torch.from_numpy(w_flat[off_ans: off_ans + alen].copy())
        seqs.append(seq); pls.append(pl); masks.append(m)
        off_full += flen; off_ans += alen
    return seqs, pls, masks


# ─────────────────────────────────────────────────────────────────────────────
def worker(expert_idx: int, expert_name: str, expert_path: str, task: str,
           device: str, alpha: float, max_prompts, result_path: str,
           prompt_slice, data_dir_str: str):
    torch.cuda.set_device(device)
    data_dir = Path(data_dir_str)
    tag = f"[w{expert_idx}/{expert_name}|{prompt_slice[0]}:{prompt_slice[1]}]"
    print(f"{tag} start on {device}", flush=True)

    t0 = time.time()
    seqs, pls, masks = load_task(task, expert_name, alpha, data_dir)
    if max_prompts:
        seqs = seqs[:max_prompts]; pls = pls[:max_prompts]; masks = masks[:max_prompts]
    s, e = prompt_slice
    seqs = seqs[s:e]; pls = pls[s:e]; masks = masks[s:e]
    print(f"{tag} {len(seqs)} prompts, load={time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    expert = AutoModelForCausalLM.from_pretrained(
        resolve_path(expert_path), torch_dtype=torch.bfloat16,
        device_map={"": device}).eval()
    print(f"{tag} model loaded in {time.time()-t1:.1f}s", flush=True)

    state = {"a": 0, "b": 0, "mask": None}
    acc_dev: dict = {}
    handles, linear_names = [], []
    for name, mod in expert.named_modules():
        if isinstance(mod, nn.Linear):
            linear_names.append(name)
            def make_hook(nm):
                def hook(_m, inp, _out):
                    x = inp[0]
                    a, b = state["a"], state["b"]
                    if x.dim() == 3:
                        x_seg = x[0, a:b]
                    elif x.dim() == 2:
                        x_seg = x[a:b]
                    else:
                        return
                    contrib = (x_seg.abs().float() * state["mask"][:, None]).sum(dim=0)
                    if nm in acc_dev:
                        acc_dev[nm].add_(contrib)
                    else:
                        acc_dev[nm] = contrib
                return hook
            handles.append(mod.register_forward_hook(make_hook(name)))

    weight_total = 0.0
    t_loop = time.time()
    with torch.no_grad():
        for qi, (seq, pl, km) in enumerate(zip(seqs, pls, masks), 1):
            Lseq = int(seq.shape[0])
            if Lseq <= pl:
                continue
            state["a"] = pl - 1
            state["b"] = Lseq - 1
            state["mask"] = km.to(device=device, dtype=torch.float32, non_blocking=True)
            weight_total += float(state["mask"].sum().item())
            _ = expert(seq.unsqueeze(0).to(device, non_blocking=True), use_cache=False)
            if qi % 20 == 0 or qi == len(seqs):
                dt = time.time() - t_loop
                eta = dt / qi * (len(seqs) - qi)
                print(f"{tag} {qi}/{len(seqs)} ({dt:.0f}s, ETA {eta:.0f}s)", flush=True)

    for h in handles:
        h.remove()

    acc_cpu = {f"xabs_key::{nm}.weight": acc_dev[nm].cpu()
                for nm in linear_names if nm in acc_dev}
    with open(result_path, "wb") as f:
        pickle.dump({"acc_c": acc_cpu, "weight_total": weight_total,
                       "linear_names": linear_names, "wall": time.time() - t0}, f)
    print(f"{tag} DONE wall={time.time()-t0:.1f}s  Σw={weight_total:.2f}", flush=True)


def merge_partial_pickles(partial_paths: list, out_path: str):
    combined_acc = None
    weight_total = 0.0
    linear_names = None
    wall_max = 0.0
    for p in partial_paths:
        with open(p, "rb") as f:
            r = pickle.load(f)
        if combined_acc is None:
            combined_acc = {k: v.clone() for k, v in r["acc_c"].items()}
            linear_names = r["linear_names"]
        else:
            for k, v in r["acc_c"].items():
                if k in combined_acc:
                    combined_acc[k].add_(v)
                else:
                    combined_acc[k] = v.clone()
        weight_total += r["weight_total"]
        wall_max = max(wall_max, r.get("wall", 0.0))
    with open(out_path, "wb") as f:
        pickle.dump({"acc_c": combined_acc, "weight_total": weight_total,
                       "linear_names": linear_names, "wall": wall_max}, f)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="W_col extraction with abs+soft weighting")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--experts", nargs=3, required=True,
                     help="3 experts as 'name=path'. Task name = expert name; "
                          "data_dir/<name>.npz must exist.")
    ap.add_argument("--data_dir", required=True,
                     help="dir with <task>.npz per expert name")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--dp", type=int, default=2, help="data-parallel degree")
    ap.add_argument("--alpha", type=float, default=1.0,
                     help="exponent for |Δlogp|^α (default 1.0 = linear)")
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    setup_cache(args.cache_dir)
    if args.dp < 1:
        raise ValueError("--dp must be >= 1")
    n_gpus = torch.cuda.device_count()
    if n_gpus < args.dp:
        raise RuntimeError(f"--dp={args.dp} but only {n_gpus} CUDA device(s) visible")

    experts = parse_experts(args.experts)             # [(name, path), x3]
    devices = [f"cuda:{i}" for i in range(args.dp)]
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / f"_tmp_{out_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print("=" * 64)
    print(f"  extract_w.py  DP={args.dp}  α={args.alpha}")
    print(f"  data_dir={args.data_dir}")
    print(f"  GPUs: {devices}")
    print(f"  experts:")
    for n, p in experts:
        print(f"    {n}: {p}")
    print("=" * 64)

    t_total_start = time.time()
    final_paths = []

    for ei, (expert_name, expert_path) in enumerate(experts):
        z = np.load(Path(args.data_dir) / f"{expert_name}.npz", allow_pickle=True)
        n_total = len(z["prompt_lens"])
        if args.max_prompts:
            n_total = min(n_total, args.max_prompts)
        del z

        N = args.dp
        chunk_starts = [(n_total * i) // N for i in range(N + 1)]
        slices = [(chunk_starts[i], chunk_starts[i + 1]) for i in range(N)]
        slices = [(s, e) for s, e in slices if s < e]

        partial_paths = [str(tmp_dir / f"{expert_name}_part{i}.pkl")
                          for i in range(len(slices))]
        print(f"\n[expert {ei+1}/3] {expert_name}: prompts={n_total}, "
              f"split={[s for s in slices]}")
        t_e = time.time()
        procs = []
        for i, (sl, dev) in enumerate(zip(slices, devices)):
            p = mp.Process(target=worker, args=(ei, expert_name, expert_path,
                                                  expert_name, dev,
                                                  args.alpha, args.max_prompts,
                                                  partial_paths[i], sl, args.data_dir))
            p.start(); procs.append(p)
        for p in procs:
            p.join()
        for i, p in enumerate(procs):
            if p.exitcode != 0:
                print(f"[FATAL] expert {expert_name} shard {i} exit={p.exitcode}")
                sys.exit(1)
        expert_pkl = str(tmp_dir / f"{expert_name}.pkl")
        merge_partial_pickles(partial_paths, expert_pkl)
        final_paths.append(expert_pkl)
        print(f"[expert {ei+1}/3] done in {time.time()-t_e:.1f}s")

    t_extraction = time.time() - t_total_start

    results = []
    for path in final_paths:
        with open(path, "rb") as f:
            results.append(pickle.load(f))

    union_layers = set().union(*[set(r["acc_c"].keys()) for r in results])
    layer_keys = sorted({k.split("::", 1)[1] for k in union_layers
                          if k.startswith("xabs_key::")})

    payload = {}
    for layer_key in layer_keys:
        stack = []
        ok = True
        for r in results:
            k = f"xabs_key::{layer_key}"
            if k not in r["acc_c"]:
                ok = False; break
            arr = r["acc_c"][k].numpy() / max(r["weight_total"], 1e-12)
            stack.append(arr)
        if not ok or len(stack) != 3:
            continue
        if len({a.shape[0] for a in stack}) != 1:
            continue
        payload[layer_key] = np.stack(stack, axis=0).astype(np.float32)

    np.savez_compressed(out_path, **payload)
    print()
    print("=" * 64)
    print(f"  TOTAL extraction wall = {t_extraction:.1f}s ({t_extraction/60:.2f} min)")
    print(f"  Per-expert weight totals (Σ |Δlogp|^α):")
    for r, (name, _) in zip(results, experts):
        print(f"    {name}: Σw={r['weight_total']:.2f}  wall={r['wall']:.1f}s")
    print(f"  Output: {out_path}  ({out_path.stat().st_size/1e6:.1f}MB, "
          f"{len(payload)} layers)")
    print("=" * 64)
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
