"""Wnorm correction: multiply W_col by per-(expert, layer, col) ‖W_expert[:,c]‖₂.

Without Wnorm, the column-side weighting is asymmetric vs the row-side
(which already absorbs W via the output activation y = W·x). After Wnorm,
both sides share the same |W·x|-style form:

    ω_col[i, ℓ, c] = ‖W_expert_i,ℓ[:, c]‖₂ · W_col[i, ℓ, c]

Variant-agnostic — pass --experts NAME=PATH.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from helpers import load_state_dict  # type: ignore


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


def main():
    ap = argparse.ArgumentParser(description="Apply Wnorm correction to W_col npz")
    ap.add_argument("--experts", nargs=3, required=True,
                     help="3 experts as 'name=path' (order must match extract_w)")
    ap.add_argument("--w_col_in", required=True, help="W_col npz from extract_w.py")
    ap.add_argument("--w_col_out", required=True, help="output Wnorm-corrected npz")
    args = ap.parse_args()

    experts = parse_experts(args.experts)

    src = Path(args.w_col_in)
    out = Path(args.w_col_out)
    if not src.exists():
        raise FileNotFoundError(src)

    print(f"[load] {src}")
    W_col = dict(np.load(src, allow_pickle=True))
    print(f"  {len(W_col)} layers")

    expert_sds = []
    for name, path in experts:
        print(f"[load] expert {name}")
        expert_sds.append(load_state_dict(path))

    new_payload = {}
    n_skipped = 0
    for layer_key, arr in W_col.items():
        if arr.ndim != 2 or arr.shape[0] != 3:
            print(f"  [skip] {layer_key}: unexpected shape {arr.shape}")
            n_skipped += 1
            new_payload[layer_key] = arr
            continue

        norms = []
        bad = False
        for ei, ((name, _), sd) in enumerate(zip(experts, expert_sds)):
            if layer_key not in sd:
                print(f"  [skip] {layer_key}: missing in {name}")
                bad = True; break
            W = sd[layer_key].float()
            if W.dim() != 2:
                bad = True; break
            colnorm = W.norm(dim=0).cpu().numpy().astype(np.float32)
            if colnorm.shape[0] != arr.shape[1]:
                print(f"  [skip] {layer_key}: shape mismatch "
                       f"colnorm={colnorm.shape} vs arr_d_in={arr.shape[1]}")
                bad = True; break
            norms.append(colnorm)
        if bad:
            n_skipped += 1
            new_payload[layer_key] = arr
            continue

        norms = np.stack(norms, axis=0)
        new_payload[layer_key] = (arr * norms).astype(np.float32)

    np.savez_compressed(out, **new_payload)
    print(f"[save] {out}  ({out.stat().st_size/1e6:.1f}MB)  "
          f"{len(new_payload)} layers  ({n_skipped} skipped)")

    sample_key = next(iter(W_col))
    if sample_key in new_payload and new_payload[sample_key].ndim == 2:
        old = W_col[sample_key]; new = new_payload[sample_key]
        if not np.allclose(old, new):
            r = (new / np.maximum(old, 1e-12)).mean()
            print(f"[sanity] {sample_key}  mean ratio new/old = {r:.3f}")


if __name__ == "__main__":
    main()
