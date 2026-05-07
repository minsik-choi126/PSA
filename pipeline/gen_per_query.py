"""Generate per_query npz files for the W extraction step.

For each task t and its associated chat-format prompts (jsonl with {"chat": [
    {"role": "user", "content": "..."}, ...], "source": "..."}):

    1. Roll out from BASE   :  base.generate(prompt) → answer tokens y_t
    2. Teacher-force base   :  log p_base(y_t)
    3. Teacher-force expert : log p_E(y_t) for each expert in --experts

Output layout matches what extract_w.py reads:

    full_tokens    (Σ full_seq_lens,) int32
    prompt_lens    (N,)               int32
    full_seq_lens  (N,)               int32
    seq_lens       (N,)               int32
    expert_names   (E,)               str
    base_lp        (T,)               float32
    expert_lp      (E, T)             float32

CLI:
    python gen_per_query.py \\
        --base_model PATH \\
        --experts NAME1=PATH1 NAME2=PATH2 NAME3=PATH3 \\
        --prompts_dir prompts/ \\
        --out_dir per_query/ \\
        [--max_ctx 4096] [--max_new_tokens 256] [--temperature 0.7]

`prompts_dir/<name>.jsonl` must exist for each expert name.
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from helpers import resolve_path, setup_cache  # type: ignore


def parse_experts(args_list):
    out = []
    for s in args_list:
        if "=" not in s:
            raise ValueError(f"--experts entry must be 'name=path'; got {s!r}")
        name, path = s.split("=", 1)
        out.append((name.strip(), path.strip()))
    if len(out) != 3:
        raise ValueError(f"need exactly 3 experts (got {len(out)})")
    return out


def load_chat_prompts(jsonl_path: Path):
    chats, sources = [], []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            chats.append(d["chat"])
            sources.append(d.get("source", "unknown"))
    return chats, sources


def apply_chat_template(tokenizer, chat, max_ctx: int):
    """Format chat using tokenizer's chat_template; head-truncate to max_ctx."""
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: simple concatenation
        parts = []
        for m in chat:
            parts.append(f"<|{m['role']}|>\n{m['content']}\n")
        parts.append("<|assistant|>\n")
        text = "".join(parts)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.numel() > max_ctx:
        ids = ids[-max_ctx:]
    return ids


def teacher_force_lp(model, prompt_ids: torch.Tensor, full_ids: torch.Tensor,
                       device: str) -> np.ndarray:
    """log p(y_t | prefix) for each answer token y_t. Returns (ans_len,) float32."""
    full = full_ids.to(device).unsqueeze(0)
    with torch.no_grad():
        logits = model(full, use_cache=False).logits[0]    # (Lfull, V) bf16/fp32
    # log p(y_t) is at position (pl-1+t) in logits, since logits[i] predicts token at i+1
    pl = prompt_ids.numel()
    Lfull = full_ids.numel()
    ans_len = Lfull - pl
    if ans_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    # logits at positions [pl-1, pl, ..., Lfull-2] predict full[pl..Lfull-1]
    ans_logits = logits[pl - 1: Lfull - 1].float()         # (ans_len, V)
    ans_targets = full_ids[pl: Lfull].to(device).long()    # (ans_len,)
    log_probs = torch.log_softmax(ans_logits, dim=-1)
    lp = log_probs.gather(1, ans_targets.unsqueeze(1)).squeeze(1)
    return lp.cpu().numpy().astype(np.float32)


def run_task(task: str, expert_paths_named: list, base_model_path: str,
              prompts_path: Path, out_path: Path, n_queries: int,
              max_ctx: int, max_new_tokens: int, temperature: float,
              device: str, seed: int):
    """One task: rollout + teacher-force scoring → npz."""
    print(f"\n══ task={task}   prompts={prompts_path}   out={out_path}")
    chats, sources = load_chat_prompts(prompts_path)
    rng = np.random.default_rng(seed)
    if n_queries < len(chats):
        idx = rng.permutation(len(chats))[:n_queries]
    else:
        idx = np.arange(len(chats))
    chats = [chats[int(i)] for i in idx]
    sources = [sources[int(i)] for i in idx]
    print(f"  sampled {len(chats)} prompts (seed={seed})")

    # ── Base model rollout ──
    print(f"  loading base for rollout: {base_model_path}")
    tok = AutoTokenizer.from_pretrained(resolve_path(base_model_path),
                                          trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        resolve_path(base_model_path), torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    ).eval()

    full_tokens_list = []
    prompt_lens = []
    full_seq_lens = []
    seq_lens = []

    print(f"  rolling out from base (max_new={max_new_tokens}, temp={temperature})")
    t0 = time.time()
    for qi, chat in enumerate(chats, 1):
        prompt_ids = apply_chat_template(tok, chat, max_ctx)
        with torch.no_grad():
            gen = base.generate(
                prompt_ids.to(device).unsqueeze(0),
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                pad_token_id=tok.eos_token_id,
            )
        full_ids = gen[0].cpu()
        full_tokens_list.append(full_ids.numpy().astype(np.int32))
        prompt_lens.append(int(prompt_ids.numel()))
        full_seq_lens.append(int(full_ids.numel()))
        seq_lens.append(int(full_ids.numel() - prompt_ids.numel()))
        if qi % 16 == 0 or qi == len(chats):
            print(f"    rollout {qi}/{len(chats)}  ({time.time()-t0:.0f}s)")

    # Score base on rollouts
    print("  teacher-force scoring base on rollouts")
    base_lp_concat = []
    for ids_np, pl, fl in zip(full_tokens_list, prompt_lens, full_seq_lens):
        full_ids = torch.from_numpy(ids_np.astype(np.int64))
        prompt_ids = full_ids[:pl]
        lp = teacher_force_lp(base, prompt_ids, full_ids, device)
        base_lp_concat.append(lp)

    del base
    torch.cuda.empty_cache(); gc.collect()

    # ── Per-expert teacher-force ──
    expert_names = [n for n, _ in expert_paths_named]
    expert_lp_full = []
    for name, path in expert_paths_named:
        print(f"  loading expert {name}: {path}")
        e = AutoModelForCausalLM.from_pretrained(
            resolve_path(path), torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=True,
        ).eval()
        per = []
        for ids_np, pl in zip(full_tokens_list, prompt_lens):
            full_ids = torch.from_numpy(ids_np.astype(np.int64))
            prompt_ids = full_ids[:pl]
            per.append(teacher_force_lp(e, prompt_ids, full_ids, device))
        expert_lp_full.append(per)
        del e
        torch.cuda.empty_cache(); gc.collect()

    # Flatten per-query lp arrays into 1-D / 2-D
    base_lp_flat = np.concatenate(base_lp_concat) if base_lp_concat else np.zeros(0, np.float32)
    expert_lp_flat = np.stack([
        np.concatenate(per) if per else np.zeros(0, np.float32)
        for per in expert_lp_full
    ])  # (E, T)
    full_tokens_flat = np.concatenate(full_tokens_list) if full_tokens_list else np.zeros(0, np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        full_tokens=full_tokens_flat,
        prompt_lens=np.array(prompt_lens, dtype=np.int32),
        full_seq_lens=np.array(full_seq_lens, dtype=np.int32),
        seq_lens=np.array(seq_lens, dtype=np.int32),
        expert_names=np.array(expert_names, dtype=object),
        base_lp=base_lp_flat,
        expert_lp=expert_lp_flat,
        sources=np.array(sources, dtype=object),
    )
    print(f"  saved → {out_path}  "
          f"(N={len(chats)}, T={base_lp_flat.size}, "
          f"{out_path.stat().st_size/1e6:.1f}MB)")


def main():
    ap = argparse.ArgumentParser(description="Generate per_query npz from chat prompts")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--experts", nargs=3, required=True,
                     help="3 experts as 'name=path'. <name>.jsonl must be in --prompts_dir.")
    ap.add_argument("--prompts_dir", required=True,
                     help="dir with <name>.jsonl chat prompt files (one per expert)")
    ap.add_argument("--out_dir", required=True,
                     help="output dir; <name>.npz written per expert")
    ap.add_argument("--n_queries", type=int, default=128)
    ap.add_argument("--max_ctx", type=int, default=4096,
                     help="default per-task max prompt context (head-truncated)")
    ap.add_argument("--max_ctx_per_task", default="",
                     help="comma-separated overrides, e.g. 'lucy=8192,memory=32768'")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    setup_cache(args.cache_dir)
    experts = parse_experts(args.experts)

    per_task_ctx = {}
    if args.max_ctx_per_task:
        for kv in args.max_ctx_per_task.split(","):
            kv = kv.strip()
            if not kv: continue
            k, v = kv.split("=")
            per_task_ctx[k.strip()] = int(v.strip())

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = Path(args.prompts_dir)

    print("=" * 64)
    print(f"  gen_per_query.py   {time.strftime('%F %T')}")
    print(f"  base : {args.base_model}")
    for n, p in experts:
        print(f"  {n}: {p}")
    print(f"  prompts_dir: {prompts_dir}")
    print(f"  out_dir    : {out_dir}")
    print(f"  n_queries={args.n_queries}  max_ctx={args.max_ctx}  "
          f"max_new={args.max_new_tokens}  temp={args.temperature}")
    print("=" * 64)

    for ei, (name, _) in enumerate(experts):
        prompts_path = prompts_dir / f"{name}.jsonl"
        if not prompts_path.exists():
            raise FileNotFoundError(prompts_path)
        out_path = out_dir / f"{name}.npz"
        ctx = per_task_ctx.get(name, args.max_ctx)
        run_task(
            task=name,
            expert_paths_named=experts,
            base_model_path=args.base_model,
            prompts_path=prompts_path, out_path=out_path,
            n_queries=args.n_queries, max_ctx=ctx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, device=args.device,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
