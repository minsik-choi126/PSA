#!/usr/bin/env python3
"""
Unified Model Merging Script

A unified script for model merging — just select model paths and a merging method.

Supported merging methods (--method):
  task_arithmetic   : Task Arithmetic (ICLR 2023) — base + Σ λ_i·(expert_i - base)  [default λ_i = 1/N]
  ties              : TIES-Merging (NeurIPS 2023) — Trim, Elect Sign & Disjoint Merge
  dare              : DARE (ICML 2024) — Drop And REscale + task_arithmetic or ties
  star              : STAR (NAACL 2025) — Singular value Truncation And Rescaling
  cart              : CART (arXiv 2024) — Centered And Rank-Truncated
  tsv               : TSV (CVPR 2025) — Task Singular Vectors (SVD-based)
  fisher            : Fisher Merging (NeurIPS 2022) — Fisher precision-weighted merge
  iso_c             : Iso-C (ICML 2025) — Isotropic Merging in Common Subspace
  iso_cts           : Iso-CTS (ICML 2025) — Isotropic Merging in Common & Task-Specific Subspaces
  ram               : RAM (arXiv 2026) — Reinforced Agentic Merge, overlap-aware averaging
  ram_plus          : RAM+ (arXiv 2026) — RAM with overlap-aware rescaling

Basic usage:
  python merge_opensource.py --method task_arithmetic \\
      --base_model  Qwen/Qwen2.5-7B-Instruct \\
      --expert_models /path/to/model_a /path/to/model_b \\
      --save_dir    /path/to/output

"""

import argparse
import gc
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Common Utilities
# ══════════════════════════════════════════════════════════════════════════════

def setup_cache(cache_dir: Optional[str] = None):
    """Set HuggingFace cache directory."""
    if cache_dir:
        hf_home = cache_dir
    else:
        # Prefer existing env var; fallback to default
        candidates = [
            os.environ.get("HF_HOME"),
            os.path.expanduser("~/.cache/huggingface"),
        ]
        hf_home = next((c for c in candidates if c and os.path.isdir(os.path.dirname(c))), None)
        if hf_home is None:
            return  # proceed without cache config

    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))


def _resolve_path(model_path: str) -> str:
    """Return local dir as-is, or resolve from HF cache / download from Hub."""
    if os.path.isdir(model_path):
        return model_path
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.path.join(hf_home, "hub")
    slug = "models--" + model_path.replace("/", "--")
    snapshots = os.path.join(hub_dir, slug, "snapshots")
    if os.path.isdir(snapshots):
        snaps = sorted(Path(snapshots).iterdir(), key=lambda p: p.stat().st_mtime)
        if snaps:
            return str(snaps[-1])
    return model_path


def load_state_dict(model_path: str) -> dict:
    """Load state dict: prefer safetensors, fall back to HF AutoModel (float32, CPU)."""
    resolved = _resolve_path(model_path)
    sf_files = sorted(Path(resolved).glob("*.safetensors"))
    if sf_files:
        from safetensors.torch import load_file
        print(f"  Loading safetensors ({len(sf_files)} shards): {resolved}")
        sd = {}
        for sf in sf_files:
            sd.update(load_file(str(sf), device="cpu"))
        return sd

    print(f"  Loading via HF AutoModel: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    sd = model.state_dict()
    del model
    gc.collect()
    return sd


def save_model(base_model_path: str, merged_sd: dict, out_dir: str):
    """Save in HuggingFace format (safetensors + tokenizer)."""
    print(f"  Saving → {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.load_state_dict(merged_sd, strict=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    del model
    gc.collect()
    tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tok.save_pretrained(out_dir)


def get_numeric_keys(base_sd: dict, expert_sds: List[dict]) -> List[str]:
    """Extract common numeric keys across all state dicts (excluding int64, uint8)."""
    common = set(base_sd.keys())
    for sd in expert_sds:
        common &= set(sd.keys())
    return sorted(
        k for k in common
        if base_sd[k].dtype not in (torch.int64, torch.uint8)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Merging Method Implementations
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# 1. Task Arithmetic (ICLR 2023)
# ──────────────────────────────────────────────────────────────────────────────

def run_task_arithmetic(base_sd: dict, expert_sds: List[dict], lambdas: List[float]) -> dict:
    """
    base + Σ λ_i · (expert_i - base)
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    final_sd = {}

    with torch.no_grad():
        for key in tqdm(keys, desc="Task Arithmetic"):
            merged = base_sd[key].float().clone()
            for i, sd in enumerate(expert_sds):
                merged.add_(lambdas[i] * (sd[key].float() - base_sd[key].float()))
            final_sd[key] = merged

    # Copy non-numeric keys from base
    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 2. TIES Merging (NeurIPS 2023)
# ──────────────────────────────────────────────────────────────────────────────

def _state_dict_to_vector(sd: dict, remove_keys: set) -> torch.Tensor:
    items = sorted((k, v) for k, v in sd.items() if k not in remove_keys)
    # Cast to float32: prevent mixed-dtype cat when loading bfloat16/float16 safetensors
    return torch.nn.utils.parameters_to_vector(
        [v.float().reshape(-1) for _, v in items]
    )


def _vector_to_state_dict(vector: torch.Tensor, ref_sd: dict, remove_keys: set) -> dict:
    ref = OrderedDict(sorted((k, v.clone()) for k, v in ref_sd.items() if k not in remove_keys))
    torch.nn.utils.vector_to_parameters(vector, ref.values())
    return ref


def _topk_mask(M: torch.Tensor, K: float) -> torch.Tensor:
    """Keep only top K% values per row by magnitude."""
    if K > 1:
        K /= 100
    if M.dim() == 1:
        M = M.unsqueeze(0)
    n, d = M.shape
    k = max(1, d - int(d * K))
    kth, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    return M * (M.abs() >= kth)


def _resolve_sign(T: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mass":
        signs = torch.sign(T.sum(dim=0))
    elif method == "normfrac":
        norms = torch.norm(T, dim=1, keepdim=True)
        nf = T ** 2 / (norms ** 2 + 1e-12)
        signs = torch.sign(T[nf.argmax(dim=0), torch.arange(T.shape[1])])
    elif method == "normmass":
        norms = torch.norm(T, dim=1, keepdim=True)
        nf = T ** 2 / (norms ** 2 + 1e-12)
        signs = (T.sign() * nf.abs()).sum(dim=0).sign()
    else:
        raise ValueError(f"Unknown sign_method: {method}")
    # Fill zero positions with overall majority sign
    majority = torch.sign(signs.sum())
    signs[signs == 0] = majority
    return signs


def _disjoint_merge(T: torch.Tensor, signs: torch.Tensor, func: str) -> torch.Tensor:
    agg = func.split("-")[-1]
    mask = torch.where(signs.unsqueeze(0) > 0, T > 0, T < 0)
    selected = T * mask
    if agg == "mean":
        count = (selected != 0).sum(dim=0).float()
        return selected.sum(dim=0) / count.clamp(min=1)
    elif agg == "sum":
        return selected.sum(dim=0)
    elif agg == "max":
        return selected.abs().max(dim=0)[0] * signs
    else:
        raise ValueError(f"Unknown merge_func: {func}")


def run_ties(
    base_sd: dict, expert_sds: List[dict],
    lamda: float, density: float, sign_method: str, merge_func: str,
    device: str,
) -> dict:
    """
    Trim → Elect Sign → Disjoint Merge → base + λ·merged_tv
    """
    int_keys = {k for k in base_sd if base_sd[k].dtype in (torch.int64, torch.uint8)}

    flat_base = _state_dict_to_vector(base_sd, int_keys).to(device)
    flat_experts = torch.vstack([
        _state_dict_to_vector(sd, int_keys).to(device) for sd in expert_sds
    ])

    tvs = flat_experts - flat_base
    del flat_experts

    print(f"  TRIM: top-{density*100:.0f}%")
    trimmed = _topk_mask(tvs, density)
    del tvs

    print(f"  ELECT SIGN: {sign_method}")
    signs = _resolve_sign(trimmed, sign_method)

    print(f"  DISJOINT MERGE: {merge_func}")
    if "dis" in merge_func:
        merged_tv = _disjoint_merge(trimmed, signs, merge_func)
    elif merge_func == "sum":
        merged_tv = trimmed.sum(dim=0)
    else:  # "mean"
        merged_tv = trimmed.mean(dim=0)

    del trimmed

    final_flat = flat_base + lamda * merged_tv
    del flat_base, merged_tv

    final_sd = _vector_to_state_dict(final_flat.cpu(), base_sd, int_keys)
    del final_flat

    for k in int_keys:
        final_sd[k] = base_sd[k]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 3. DARE (ICML 2024) — Drop And REscale Merging
# ──────────────────────────────────────────────────────────────────────────────

def _dare_mask_tensor(
    delta: torch.Tensor,
    mask_rate: float,
    use_rescale: bool,
    mask_strategy: str,
) -> torch.Tensor:
    """
    DARE core operation: drop mask_rate fraction of elements in delta tensor,
    then rescale by 1 / (1 - mask_rate) if use_rescale=True.

    mask_strategy:
      "random"    : Random drop via Bernoulli sampling
      "magnitude" : Drop lowest mask_rate fraction by absolute value
    """
    assert 0.0 <= mask_rate <= 1.0, f"mask_rate must be in [0, 1], got {mask_rate}"

    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(delta, mask_rate))
        masked = delta * (1 - mask)
    elif mask_strategy == "magnitude":
        original_shape = delta.shape
        flat = delta.flatten()
        num_mask = int(len(flat) * mask_rate)
        if num_mask == 0:
            return delta.clone()
        kth_val, _ = flat.abs().kthvalue(k=num_mask, dim=0, keepdim=True)
        keep_mask = flat.abs() >= kth_val
        masked = (flat * keep_mask).reshape(original_shape)
    else:
        raise ValueError(f"Unknown mask_strategy: {mask_strategy}")

    if use_rescale and mask_rate < 1.0:
        masked = masked / (1 - mask_rate)

    return masked


def run_dare(
    base_sd: dict,
    expert_sds: List[dict],
    lambdas: List[float],
    weight_mask_rate: float,
    use_rescale: bool,
    mask_strategy: str,
    merge_method: str,
        # TIES-specific params (used only when merge_method == "ties")
    ties_lamda: float = 1.0,
    ties_density: float = 0.2,
    ties_sign_method: str = "mass",
    ties_merge_func: str = "dis-mean",
    device: str = "cpu",
) -> dict:
    """
    DARE (Drop And REscale) Merging.
    Reference: https://github.com/yule-BUAA/MergeLM

    Step 1: Apply _dare_mask_tensor to each expert's delta = expert - base
            (drop mask_rate fraction + optional rescaling)
    Step 2: Merge sparsified experts via task_arithmetic or ties

    Args:
        base_sd          : Base model state dict
        expert_sds       : List of expert model state dicts
        lambdas          : Per-expert scaling coefficients (for task_arithmetic)
        weight_mask_rate : Fraction of delta elements to drop (0.0-1.0, paper default 0.9)
        use_rescale      : If True, rescale surviving elements by 1/(1-mask_rate)
        mask_strategy    : "random" | "magnitude"
        merge_method     : "task_arithmetic" | "ties"
        ties_*           : TIES params (only used when merge_method=="ties")
        device           : TIES compute device
    """
    keys = get_numeric_keys(base_sd, expert_sds)

    # -- Step 1: DARE sparsification --
    sparsified_sds = []
    for i, expert_sd in enumerate(expert_sds):
        print(f"  [DARE] Expert {i}: mask_rate={weight_mask_rate}, "
              f"strategy={mask_strategy}, rescale={use_rescale}")
        sparse_sd = {}
        with torch.no_grad():
            for key in tqdm(keys, desc=f"    Expert {i} DARE"):
                delta = expert_sd[key].float() - base_sd[key].float()
                sparse_delta = _dare_mask_tensor(delta, weight_mask_rate, use_rescale, mask_strategy)
                sparse_sd[key] = base_sd[key].float() + sparse_delta
        # Copy non-numeric keys as-is
        for key in expert_sd:
            if key not in sparse_sd:
                sparse_sd[key] = expert_sd[key]
        sparsified_sds.append(sparse_sd)

    # -- Step 2: Merge using sparsified experts --
    print(f"  [DARE] Post-merge method: {merge_method}")
    if merge_method == "task_arithmetic":
        return run_task_arithmetic(base_sd, sparsified_sds, lambdas)
    elif merge_method == "ties":
        return run_ties(
            base_sd, sparsified_sds,
            ties_lamda, ties_density, ties_sign_method, ties_merge_func, device,
        )
    else:
        raise ValueError(f"Unknown merge_method for DARE: {merge_method}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. STAR (NAACL 2025) — Singular value Truncation And Rescaling
# ──────────────────────────────────────────────────────────────────────────────

def _star_compress_tensor(delta: torch.Tensor, eta: float) -> torch.Tensor:
    """
    STAR core operation: SVD of a 2D delta matrix, truncate to minimum rank
    covering eta% of nuclear norm, then rescale remaining singular values to preserve total sum.

    Formulation:
      sum_tot = Σ sᵢ
      rank_remain = min r s.t. Σᵢ₌₁ʳ sᵢ ≥ eta/100 · sum_tot
      scaled_s = (sum_tot / Σᵢ₌₁ʳ sᵢ) · s[:r]
      output = U[:, :r] @ diag(scaled_s) @ Vt[:r, :]

    1D tensors (bias, layernorm, etc.) are returned as-is.
    """
    if delta.dim() < 2:
        return delta.clone()

    u, s, vt = torch.linalg.svd(delta, full_matrices=False)

    sum_tot_s = torch.sum(s)
    if sum_tot_s == 0:
        return delta.clone()

    # Find minimum rank covering eta% of nuclear norm
    cumulative_s = torch.cumsum(s, dim=0)
    rank_remain = torch.searchsorted(cumulative_s, sum_tot_s * eta / 100.0).item() + 1
    rank_remain = max(1, min(rank_remain, s.shape[0]))

    # Truncate
    u_t  = u[:, :rank_remain]
    s_t  = s[:rank_remain]
    vt_t = vt[:rank_remain, :]

    # Rescale: Σ scaled_sᵢ = sum_tot_s
    sum_remain = torch.sum(s_t)
    scaled_s = (sum_tot_s / sum_remain) * s_t

    return u_t @ torch.diag(scaled_s) @ vt_t


def run_star(
    base_sd: dict,
    expert_sds: List[dict],
    eta: float,
    lambdas: List[float],
    device: str = "cpu",
) -> dict:
    """
    STAR (Singular value Truncation And Rescaling) Merging.
    Reference: https://github.com/IBM/STAR (NAACL 2025)

    Step 1: Apply _star_compress_tensor to each expert's delta
            (truncate to rank covering eta% nuclear norm + singular value rescale)
    Step 2: Weighted average of STAR-compressed deltas, then add to base

    Args:
        base_sd    : Base model state dict
        expert_sds : List of expert model state dicts
        eta        : Nuclear norm retention ratio (%, default 40). Lower = more compression
        lambdas    : Per-expert scaling coefficients (paper default: [1/N, ...] uniform)
        device     : Device for SVD computation (cpu or cuda:N). cpu is ~20x slower.
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    N = len(expert_sds)

    # -- Step 1: STAR compression on each expert delta --
    star_deltas = []
    for i, expert_sd in enumerate(expert_sds):
        print(f"  [STAR] Expert {i}: eta={eta}, device={device}")
        compressed = {}
        with torch.no_grad():
            for key in tqdm(keys, desc=f"    Expert {i} STAR SVD"):
                delta = (expert_sd[key].float().to(device)
                         - base_sd[key].float().to(device))
                compressed[key] = _star_compress_tensor(delta, eta).cpu()
                del delta
        star_deltas.append(compressed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- Step 2: Lambda-weighted average + add to base --
    final_sd = {}
    with torch.no_grad():
        for key in tqdm(keys, desc="  STAR merge"):
            merged_delta = torch.zeros_like(base_sd[key].float())
            for i in range(N):
                merged_delta.add_(lambdas[i] * star_deltas[i][key])
            final_sd[key] = base_sd[key].float() + merged_delta

    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 5. CART (arXiv 2024, arXiv:2412.12153) — Centered And Rank-Truncated
# ──────────────────────────────────────────────────────────────────────────────

def _cart_lowrank_tensor(delta: torch.Tensor, rank_ratio: float) -> torch.Tensor:
    """Apply low-rank approximation to a 2D tensor (CART SVD truncation)."""
    u, s, vt = torch.linalg.svd(delta, full_matrices=False)
    min_dim = s.shape[0]
    rank = max(1, int(rank_ratio * min_dim))
    rank = min(rank, min_dim)
    return u[:, :rank] @ torch.diag(s[:rank]) @ vt[:rank, :]


def run_cart(
    base_sd: dict, expert_sds: List[dict],
    prior: float, rank_ratio: float,
    device: str = "cpu",
) -> dict:
    """
    CART: Centered And Rank-Truncated merge (Preprint, arXiv:2412.12153).

    1. theta_avg  = base + (1/N) · Σ delta_i          (weight average)
    2. c_delta_i  = expert_i - theta_avg               (centering)
    3. low-rank approx on 2D layers via SVD truncation  (rank = rank_ratio · min_dim)
    4. merged     = theta_avg + prior · Σ trunc(c_delta_i)

    Recommended hyperparameters from the paper (ViT-B/32, 8/14/20 tasks):
      prior      : 2.0 / 1.5 / 1.9
      rank_ratio : 0.12 / 0.16 / 0.32

    device : where SVD runs. cpu is ~20x slower than cuda for large 2D layers.
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    N = len(expert_sds)

    # -- Step 1: theta_avg --
    print(f"  [CART] Step 1: computing weight average (theta_avg)")
    theta_avg = {}
    with torch.no_grad():
        for key in keys:
            base = base_sd[key].float()
            delta_sum = torch.zeros_like(base)
            for expert_sd in expert_sds:
                delta_sum.add_(expert_sd[key].float() - base)
            theta_avg[key] = base + delta_sum / N

    # -- Step 2 & 3: centered deltas + low-rank approximation --
    print(f"  [CART] Step 2-3: centered deltas + SVD low-rank (rank_ratio={rank_ratio}, device={device})")
    lowrank_sum = {}
    with torch.no_grad():
        for key in tqdm(keys, desc="  CART SVD"):
            apply_svd = (base_sd[key].dim() == 2)
            theta_dev = theta_avg[key].to(device) if apply_svd else theta_avg[key]
            centered_sum = torch.zeros_like(theta_dev)
            for expert_sd in expert_sds:
                if apply_svd:
                    c_delta = expert_sd[key].float().to(device) - theta_dev
                    c_delta = _cart_lowrank_tensor(c_delta, rank_ratio)
                else:
                    c_delta = expert_sd[key].float() - theta_avg[key]
                centered_sum.add_(c_delta)
            lowrank_sum[key] = centered_sum.cpu() if apply_svd else centered_sum
            del centered_sum
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- Step 4: final merge --
    final_sd = {}
    with torch.no_grad():
        for key in keys:
            final_sd[key] = theta_avg[key] + prior * lowrank_sum[key]

    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 6. TSV (CVPR 2025) — Task Singular Vectors
# ──────────────────────────────────────────────────────────────────────────────

def run_tsv(
    base_sd: dict, expert_sds: List[dict],
    alpha: float, k: Optional[int], sv_reduction: Optional[float],
    device: str,
) -> dict:
    """
    SVD-based orthogonalization and reconstruction.
    2D layers: block-diagonal SVD -> polar orthogonalization -> reconstruction
    1D layers: rolling mean
    """
    N = len(expert_sds)
    keys = get_numeric_keys(base_sd, expert_sds)

    if k is None:
        sv_reduction = sv_reduction or (1.0 / N)

    merged_tv = {}
    with torch.no_grad():
        for key in tqdm(keys, desc="TSV Merge"):
            tv_list = [sd[key].float().to(device) - base_sd[key].float().to(device)
                       for sd in expert_sds]
            shape = tv_list[0].shape

            if len(shape) >= 2:
                # 2D: SVD block-diagonal
                M, D = shape[0], shape[1]
                min_dim = min(M, D)

                # Determine k_per from first expert's SVD dimensions
                U0, S0, Vh0 = torch.linalg.svd(tv_list[0], full_matrices=False)
                k_per = min(k, min_dim) if k else max(1, int(min_dim * sv_reduction))
                if N * k_per > min_dim:
                    k_per = max(1, min_dim // N)

                sum_u = torch.zeros(M, min_dim, device=device)
                sum_s = torch.zeros(min_dim, device=device)
                sum_v = torch.zeros(min_dim, D, device=device)

                sum_u[:, :k_per] = U0[:, :k_per]
                sum_s[:k_per] = S0[:k_per]
                sum_v[:k_per, :] = Vh0[:k_per, :]
                del U0, S0, Vh0

                for i, tv in enumerate(tv_list[1:], 1):
                    U, S, Vh = torch.linalg.svd(tv, full_matrices=False)
                    sum_u[:, i * k_per:(i + 1) * k_per] = U[:, :k_per]
                    sum_s[i * k_per:(i + 1) * k_per] = S[:k_per]
                    sum_v[i * k_per:(i + 1) * k_per, :] = Vh[:k_per, :]
                    del U, S, Vh

                u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                merged_tv[key] = torch.linalg.multi_dot((
                    u_u, v_u, torch.diag(sum_s), u_v, v_v
                )).cpu()
                del sum_u, sum_s, sum_v, u_u, v_u, u_v, v_v
            else:
                # 1D: rolling mean
                result = tv_list[0].clone()
                for i, tv in enumerate(tv_list[1:], 1):
                    result.add_((tv - result) / (i + 1))
                merged_tv[key] = result.cpu()

    final_sd = {}
    for key in base_sd:
        if key in merged_tv:
            final_sd[key] = base_sd[key].float() + alpha * merged_tv[key]
        else:
            final_sd[key] = base_sd[key]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 7. Fisher Merging (NeurIPS 2022) — Precision-Weighted Merge
# ──────────────────────────────────────────────────────────────────────────────

def _compute_empirical_fisher(
    model_path: str, device: str, calib_samples: int, calib_seqlen: int, seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Empirical Fisher diagonal estimation (mean of squared gradients)."""
    print(f"\n  Computing Fisher: {model_path}")
    torch.manual_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable()
    vocab_size = model.config.vocab_size

    # Use the device of the first parameter as input device for device_map="auto"
    first_device = next(model.parameters()).device

    params = [(n, p) for n, p in model.named_parameters() if p.data.is_floating_point()]
    for _, p in params:
        p.requires_grad_(True)

    fisher = {n: torch.zeros_like(p, dtype=torch.float32, device="cpu") for n, p in params}

    for s in tqdm(range(calib_samples), desc="  Fisher estimation"):
        input_ids = torch.randint(0, vocab_size, (1, calib_seqlen), device=first_device)
        model.zero_grad()
        logits = model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        seq_lp = log_probs.gather(2, input_ids[:, 1:].unsqueeze(2)).sum()
        seq_lp.backward()
        for n, p in params:
            if p.grad is not None:
                fisher[n].add_(p.grad.detach().float().cpu().pow(2))
        del logits, log_probs, seq_lp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for n in fisher:
        fisher[n].div_(calib_samples)

    del model, params
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fisher


def run_fisher(
    base_sd: dict, expert_sds: List[dict], expert_paths: List[str],
    lambdas: List[float], epsilon: float,
    fisher_files: Optional[List[str]], calib_samples: int, calib_seqlen: int,
    fisher_device: str, save_fisher: bool, save_dir: str,
) -> dict:
    """
    Fisher precision-weighted merge (delta-space):
    θ* = base + Σ(λ_i·F_i·Δ_i) / (Σ(λ_i·F_i) + ε)
    """
    # Load or compute Fisher
    fishers = []
    if fisher_files:
        for fp in fisher_files:
            print(f"  Loading Fisher: {fp}")
            fishers.append(torch.load(fp, map_location="cpu"))
    else:
        fisher_save_dir = os.path.join(save_dir, "fisher_diagonals") if save_fisher else None
        if fisher_save_dir:
            os.makedirs(fisher_save_dir, exist_ok=True)
        for i, path in enumerate(expert_paths):
            # Fully release residual memory from previous model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            f = _compute_empirical_fisher(path, fisher_device, calib_samples, calib_seqlen, 42 + i)
            fishers.append(f)
            if fisher_save_dir:
                torch.save(f, os.path.join(fisher_save_dir, f"fisher_{i}.pt"))

    # Common keys
    common = set(base_sd.keys())
    for sd in expert_sds:
        common &= set(sd.keys())
    for f in fishers:
        common &= set(f.keys())
    keys = sorted(k for k in common if base_sd[k].dtype not in (torch.int64, torch.uint8))

    N = len(expert_sds)
    final_sd = {}
    zero_count = 0

    with torch.no_grad():
        for key in tqdm(keys, desc="Fisher Merge"):
            base = base_sd[key].float()
            num = torch.zeros_like(base)
            den = torch.zeros_like(base)
            for i in range(N):
                delta = expert_sds[i][key].float() - base
                prec = lambdas[i] * fishers[i][key].float()
                num.add_(prec * delta)
                den.add_(prec)
            zero_mask = den <= epsilon
            merged = base + num / (den + epsilon)
            merged = torch.where(zero_mask, base, merged)
            final_sd[key] = merged
            zero_count += int(zero_mask.sum().item())

    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    print(f"  Zero-precision params: {zero_count} → base preserved")
    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 8. Iso-C (ICML 2025) — Isotropic Merging in Common Subspace
# ──────────────────────────────────────────────────────────────────────────────

def run_iso_c(
    base_sd: dict, expert_sds: List[dict],
    alpha: float, device: str,
) -> dict:
    """
    Iso-C: Isotropic Merging in Common Subspace.
    Reference: "No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces"
           https://github.com/danielm1405/iso-merging (ICML 2025)

    For each 2D layer:
      1. Sum task vectors: W_sum = sum_t tau_t  (Task Arithmetic sum)
      2. SVD: W_sum = U S V^T
      3. Flatten singular value spectrum: S_iso = mean(S) * ones_like(S)
      4. Reconstruct: W_iso = U @ diag(S_iso) @ V^T
      5. Apply scaling: base + alpha * W_iso

    Non-2D layers (bias, layernorm, etc.): simple average of task vectors.
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    N = len(expert_sds)
    final_sd = {}

    with torch.no_grad():
        for key in tqdm(keys, desc="Iso-C"):
            tvs = [sd[key].float().to(device) - base_sd[key].float().to(device)
                   for sd in expert_sds]

            if base_sd[key].dim() == 2:
                # 2D: SVD on summed task vectors, then isotropic flattening
                summed_tv = sum(tvs)
                U, S, V = torch.linalg.svd(summed_tv, full_matrices=False)
                S_iso = torch.ones_like(S) * S.mean()
                merged_tv = torch.linalg.multi_dot((U, torch.diag(S_iso), V))
            else:
                # 1D: average task vectors
                merged_tv = sum(tvs) / N

            final_sd[key] = (base_sd[key].float().to(device) + alpha * merged_tv).cpu()

    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 9. Iso-CTS (ICML 2025) — Isotropic Merging in Common & Task-Specific Subspaces
# ──────────────────────────────────────────────────────────────────────────────

def run_iso_cts(
    base_sd: dict, expert_sds: List[dict],
    alpha: float, common_space_fraction: float, device: str,
) -> dict:
    """
    Iso-CTS: Isotropic Merging in Common and Task-Specific Subspaces.
    Reference: "No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces"
           https://github.com/danielm1405/iso-merging (ICML 2025)

    For each 2D layer:
      1. Common subspace: top k_c singular directions of Σ_t τ_t via SVD
      2. Task-specific subspace: top k_ts singular directions of each τ_t's
         residual after removing common directions (T tasks x k_ts dims)
      3. Concatenate task-specific block (front) + common block (back)
      4. Re-orthogonalize U, V via SVD polar decomposition
      5. Flatten singular value spectrum (mean -> isotropic)
      6. Reconstruct and apply alpha scaling

    Non-2D layers: task vector running average.

    Args:
        common_space_fraction: Fraction of singular dimensions for common subspace
                               (default: 0.8, paper recommended)
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    N = len(expert_sds)
    final_sd = {}

    with torch.no_grad():
        for key in tqdm(keys, desc="Iso-CTS"):
            shape_ = base_sd[key].shape
            is_2d_matrix = (base_sd[key].dim() == 2)

            # -- Non-2D: running average of task vectors --
            if not is_2d_matrix:
                result = None
                for i, sd in enumerate(expert_sds):
                    tv = sd[key].float().to(device) - base_sd[key].float().to(device)
                    if i == 0:
                        result = tv.clone()
                    else:
                        result = result + (tv - result) / (i + 1)
                final_sd[key] = (base_sd[key].float().to(device) + alpha * result).cpu()
                continue

            # -- 2D: Iso-CTS procedure --

            # Step 1: Sum task vectors (common subspace is based on TA sum)
            tvs = [sd[key].float().to(device) - base_sd[key].float().to(device)
                   for sd in expert_sds]
            combined_w = sum(tvs)

            min_dim = min(shape_)

            # Step 2: Determine common / task-specific split sizes
            #   - Adjust so task-specific total is a multiple of N
            common_space_index_s = int(min_dim * common_space_fraction)
            _task_specific_total = round(
                (min_dim - common_space_index_s) / N
            ) * N
            common_space_index_s = min_dim - _task_specific_total

            n_dims_per_task = int((min_dim - common_space_index_s) / N)

            # Step 3: SVD of combined_w -> common subspace (top singular directions)
            u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
            common_space_u = u[:, :common_space_index_s]   # [M, k_c]
            common_space_s = s[:common_space_index_s]       # [k_c]
            common_space_v = v[:common_space_index_s, :]    # [k_c, D]

            # Step 4: Extract task-specific subspace for each task
            #   Remove common directions via orthogonal projection, then SVD of residual
            M_, D_ = shape_
            combined_space_u = torch.zeros(M_, min_dim, device=device)
            combined_space_s = torch.zeros(min_dim, device=device)
            combined_space_v = torch.zeros(min_dim, D_, device=device)

            for i, tv in enumerate(tvs):
                # Remove common subspace component (orthogonal projection)
                w_ts = tv - common_space_u @ (common_space_u.T @ tv)

                # SVD of residual (task-specific) matrix
                u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

                # Place this task's top n_dims_per_task singular components in its slot
                combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
                combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task]    = s_ts[:n_dims_per_task]
                combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]

            # Step 5: Append common subspace block at the end
            ts_end = N * n_dims_per_task
            combined_space_u[:, ts_end : ts_end + common_space_index_s] = common_space_u
            combined_space_s[ts_end : ts_end + common_space_index_s]    = common_space_s
            combined_space_v[ts_end : ts_end + common_space_index_s, :] = common_space_v

            # Step 6: Re-orthogonalize U, V via SVD polar decomposition (nearest orthogonal matrix)
            #   Orthogonality may break from concatenating task-specific + common vectors
            #   Small noise fallback for ill-conditioned matrices
            try:
                u_uu, _, v_uu = torch.linalg.svd(combined_space_u, full_matrices=False)
            except torch._C._LinAlgError:
                combined_space_u = combined_space_u + 1e-6 * torch.randn_like(combined_space_u)
                u_uu, _, v_uu = torch.linalg.svd(combined_space_u, full_matrices=False)
            try:
                u_vv, _, v_vv = torch.linalg.svd(combined_space_v, full_matrices=False)
            except torch._C._LinAlgError:
                combined_space_v = combined_space_v + 1e-6 * torch.randn_like(combined_space_v)
                u_vv, _, v_vv = torch.linalg.svd(combined_space_v, full_matrices=False)
            combined_space_u = u_uu @ v_uu
            combined_space_v = u_vv @ v_vv

            # Step 7: Flatten singular value spectrum (isotropic)
            combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()

            # Step 8: Reconstruct merged task vector
            merged_tv = torch.linalg.multi_dot((
                combined_space_u,
                torch.diag(combined_space_s),
                combined_space_v,
            ))

            final_sd[key] = (base_sd[key].float().to(device) + alpha * merged_tv).cpu()

    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 10. RAM (arXiv 2026, arXiv:2601.13572) — Reinforced Agentic Merge
#     Reference: https://github.com/xiangchi-yuan/mrl (Preprint, arXiv:2601.13572)
# ──────────────────────────────────────────────────────────────────────────────

def run_ram(
    base_sd: dict, expert_sds: List[dict],
    threshold: float = 1e-5,
    device: str = "cpu",
) -> dict:
    """
    RAM (Reinforced Agentic Merge) — basic overlap-aware averaging.

    Original function: agentic_reinforcement_merge (ram-main.py)

    Algorithm:
      1. Compute task vectors: delta_i = expert_i - base
      2. change mask: |Δ_i| > threshold
      3. Average only changed parameters: avg = sum(delta_i * mask_i) / sum(mask_i)
      4. merged = base + avg

    When using GPU, tensors are moved per-key to conserve VRAM.
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    n = len(expert_sds)

    # -- Compute task vectors (kept on CPU) --
    print(f"  Computing {n} task vectors...")
    task_vecs = []
    for sd in expert_sds:
        tv = {}
        for k in keys:
            tv[k] = sd[k].float() - base_sd[k].float()
        task_vecs.append(tv)

    # -- Merge (per-key GPU transfer) --
    merged_sd = {}
    with torch.no_grad():
        for k in tqdm(keys, desc="RAM"):
            diffs = torch.stack([tv[k] for tv in task_vecs], dim=0).to(device)  # (n, *shape)
            change_mask = (diffs.abs() > threshold)
            change_mask_f = change_mask.float()

            sum_diff = (diffs * change_mask_f).sum(dim=0)
            count = change_mask_f.sum(dim=0)

            denom = torch.clamp(count, min=1.0)
            avg_diff = sum_diff / denom
            diff_final = torch.where(count > 0, avg_diff, torch.zeros_like(sum_diff))

            merged_sd[k] = (base_sd[k].float().to(device) + diff_final).cpu()

    for k in base_sd:
        if k not in merged_sd:
            merged_sd[k] = base_sd[k]

    return merged_sd


def run_ram_plus(
    base_sd: dict, expert_sds: List[dict],
    threshold: float = 1e-5,
    rescale_factor: float = 1.05,
    device: str = "cpu",
) -> dict:
    """
    RAM+ (ARM-R-V2) — rescaling based on overlap ratio.

    Original function: agentic_reinforcement_merge_rescale_v2 (ram-main.py)

    Algorithm:
      Phase 1 — compute rescale factor:
        - For each task j:
          changed_j = number of parameters changed by task j
          overlap_j = number of those changed by 2+ tasks simultaneously
          ratio_j   = overlap_j / (changed_j - overlap_j)
          r_j       = 1.0 + (r - 1.0) * min(2, min(1.0, ratio_j))

      Phase 2 — merging:
        - Overlap region (count >= 2): average
        - Non-overlap region (count == 1): weighted sum with rescale factor
        - merged = base + diff_final

    When using GPU, tensors are moved per-key to conserve VRAM.
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    n = len(expert_sds)

    r = float(rescale_factor)
    if r <= 1.0:
        r = 1.0

    # -- Compute task vectors (kept on CPU) --
    print(f"  Computing {n} task vectors...")
    task_vecs = []
    for sd in expert_sds:
        tv = {}
        for k in keys:
            tv[k] = sd[k].float() - base_sd[k].float()
        task_vecs.append(tv)

    # -- Phase 1: Compute overlap statistics and rescale factors (per-key GPU transfer) --
    print("  Computing overlap statistics...")
    changed_counts = [0] * n
    overlap_counts = [0] * n

    for name in tqdm(keys, desc="RAM+ overlap stats"):
        diffs = torch.stack([tv[name] for tv in task_vecs], dim=0).to(device)
        change_mask = (diffs.abs() > threshold)
        sum_change = change_mask.sum(dim=0)

        overlap_any = (sum_change >= 2)

        change_flat = change_mask.view(n, -1)
        overlap_flat = overlap_any.view(-1)

        for j in range(n):
            cj = change_flat[j]
            changed_counts[j] += cj.sum().item()
            overlap_counts[j] += (cj & overlap_flat).sum().item()

    overlap_ratios = []
    rescales = []
    for j in range(n):
        denom = changed_counts[j] - overlap_counts[j]
        if denom == 0:
            ratio = 0.0
        else:
            ratio = overlap_counts[j] / denom
        overlap_ratios.append(ratio)

        rescale_j = 1.0 + (r - 1.0) * min(2, min(1.0, float(ratio)))
        rescales.append(rescale_j)

    print(f"  [OverlapAware] overlap_ratios per task: {overlap_ratios}")
    print(f"  [OverlapAware] rescale per task: {rescales}")

    # -- Phase 2: Merge with rescaling (per-key GPU transfer) --
    merged_sd = {}
    with torch.no_grad():
        for name in tqdm(keys, desc="RAM+ merge"):
            diffs = torch.stack([tv[name] for tv in task_vecs], dim=0).to(device)

            change_mask = (diffs.abs() > threshold)
            change_mask_f = change_mask.float()

            sum_diff = (diffs * change_mask_f).sum(dim=0)
            count = change_mask_f.sum(dim=0)

            denom = torch.clamp(count, min=1.0)
            avg_diff = sum_diff / denom

            zero = torch.zeros_like(sum_diff)
            non_overlap_mask = (count == 1)
            overlap_mask = (count >= 2)

            rescales_tensor = torch.tensor(
                rescales,
                dtype=diffs.dtype,
                device=diffs.device,
            ).view((n,) + (1,) * (diffs.dim() - 1))

            weighted_sum = (diffs * change_mask_f * rescales_tensor).sum(dim=0)

            diff_final = zero
            diff_final = torch.where(overlap_mask, avg_diff, diff_final)
            diff_final = torch.where(non_overlap_mask, weighted_sum, diff_final)

            merged_sd[name] = (base_sd[name].float().to(device) + diff_final).cpu()

    for k in base_sd:
        if k not in merged_sd:
            merged_sd[k] = base_sd[k]

    return merged_sd


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified Model Merging Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common arguments
    p.add_argument("--method", required=True,
        choices=["task_arithmetic", "ties", "dare", "star", "cart", "tsv", "fisher",
                 "iso_c", "iso_cts", "ram", "ram_plus"],
        help="Merging method to use")
    p.add_argument("--base_model", required=True,
        help="Base (pretrained) model path or HuggingFace name")
    p.add_argument("--expert_models", nargs="+", required=True,
        help="Fine-tuned expert model paths (2 or more)")
    p.add_argument("--save_dir", required=True,
        help="Output directory for merged model")
    p.add_argument("--cache_dir", default=None,
        help="HuggingFace cache directory (default: env var or auto-detect)")

    # Task Arithmetic / Fisher shared
    p.add_argument("--lambdas", type=float, nargs="+", default=None,
        help="[task_arithmetic/fisher] Per-expert scaling coefficients. "
             "task_arithmetic default: 1/N (uniform mean). fisher default: all 1.0.")

    # TIES
    p.add_argument("--lamda", type=float, default=1.0,
        help="[ties] Merged task vector scale (default: 1.0)")
    p.add_argument("--density", type=float, default=0.2,
        help="[ties] TRIM step keep ratio top-K%% (default: 0.2)")
    p.add_argument("--sign_method", default="mass",
        choices=["mass", "normfrac", "normmass"],
        help="[ties] Sign election method (default: mass)")
    p.add_argument("--merge_func", default="dis-mean",
        choices=["dis-mean", "dis-sum", "dis-max", "mean", "sum"],
        help="[ties] Aggregation method (default: dis-mean)")

    # DARE
    p.add_argument("--weight_mask_rate", type=float, default=0.9,
        help="[dare] Delta drop rate (default: 0.9, paper recommended)")
    p.add_argument("--use_rescale", action="store_true", default=True,
        help="[dare] Rescale surviving elements by 1/(1-mask_rate) (default: True)")
    p.add_argument("--no_rescale", dest="use_rescale", action="store_false",
        help="[dare] Disable rescaling")
    p.add_argument("--mask_strategy", default="random",
        choices=["random", "magnitude"],
        help="[dare] Drop strategy: random(Bernoulli) | magnitude(drop small) (default: random)")
    p.add_argument("--dare_merge_method", default="task_arithmetic",
        choices=["task_arithmetic", "ties"],
        help="[dare] Post-DARE merge method (default: task_arithmetic)")

    # STAR
    p.add_argument("--eta", type=float, default=40.0,
        help="[star] Nuclear norm retention ratio %% (default: 40.0, lower = more compression)")

    # CART
    p.add_argument("--prior", type=float, default=2.0,
        help="[cart] Low-rank sum scaling (default: 2.0, paper: 8-task=2.0 / 14-task=1.5 / 20-task=1.9)")
    p.add_argument("--rank_ratio", type=float, default=0.12,
        help="[cart] SVD truncation rank ratio (default: 0.12, paper: 8-task=0.12 / 14-task=0.16 / 20-task=0.32)")

    # TSV / Iso-C / Iso-CTS
    p.add_argument("--alpha", type=float, default=1.0,
        help="[tsv/iso_c/iso_cts] Merged task vector scale (default: 1.0)")
    p.add_argument("--k", type=int, default=None,
        help="[tsv] Number of singular values to keep per expert (fixed)")
    p.add_argument("--sv_reduction", type=float, default=None,
        help="[tsv] Fraction of singular values to keep (default: 1/N)")

    # Iso-CTS
    p.add_argument("--common_space_fraction", type=float, default=0.8,
        help="[iso_cts] Fraction of singular dims for common subspace (default: 0.8, paper recommended)")

    # Fisher
    p.add_argument("--epsilon", type=float, default=1e-12,
        help="[fisher] Denominator stabilization constant (default: 1e-12)")
    p.add_argument("--fisher_files", type=str, nargs="+", default=None,
        help="[fisher] Pre-computed Fisher .pt files")
    p.add_argument("--calib_samples", type=int, default=64,
        help="[fisher] Fisher estimation calibration samples (default: 64)")
    p.add_argument("--calib_seqlen", type=int, default=128,
        help="[fisher] Calibration sequence length (default: 128)")
    p.add_argument("--fisher_device", type=str, default="cuda:0",
        help="[fisher] Fisher compute device (default: cuda:0)")
    p.add_argument("--save_fisher", action="store_true",
        help="[fisher] Save computed Fisher files")

    # RAM / RAM+
    p.add_argument("--ram_threshold", type=float, default=1e-5,
        help="[ram/ram_plus] Parameter change detection threshold (default: 1e-5)")
    p.add_argument("--rescale_factor", type=float, default=1.05,
        help="[ram_plus] Overlap-based rescale factor r (default: 1.05)")

    # Compute device (TIES, TSV)
    p.add_argument("--device", type=str, default="cpu",
        help="[ties/tsv] Compute device (default: cpu)")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    N = len(args.expert_models)
    if N < 2:
        parser.error("At least 2 expert models are required.")

    setup_cache(args.cache_dir)

    print("=" * 60)
    print(f"  Method: {args.method.upper()}")
    print("=" * 60)
    print(f"  Base:     {args.base_model}")
    for i, m in enumerate(args.expert_models):
        print(f"  Expert {i}: {m}")
    print(f"  Save:     {args.save_dir}")
    print("=" * 60)

    # -- Load models --
    print("\n[1/3] Loading models...")
    base_sd = load_state_dict(args.base_model)
    expert_sds = [load_state_dict(m) for m in args.expert_models]

    os.makedirs(args.save_dir, exist_ok=True)

    # -- Merge --
    print(f"\n[2/3] Merging ({args.method})...")

    if args.method == "task_arithmetic":
        lambdas = args.lambdas or [1.0 / N] * N
        if len(lambdas) != N:
            parser.error(f"--lambdas count ({len(lambdas)}) does not match number of models ({N}).")
        print(f"  λ = {lambdas}  (default = 1/N mean)")
        final_sd = run_task_arithmetic(base_sd, expert_sds, lambdas)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "ties":
        print(f"  λ={args.lamda}, density={args.density}, sign={args.sign_method}, func={args.merge_func}")
        final_sd = run_ties(base_sd, expert_sds, args.lamda, args.density,
                            args.sign_method, args.merge_func, args.device)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "dare":
        lambdas = args.lambdas or [1.0] * N
        if len(lambdas) != N:
            parser.error(f"--lambdas count ({len(lambdas)}) does not match number of models ({N}).")
        print(f"  mask_rate={args.weight_mask_rate}, rescale={args.use_rescale}, "
              f"strategy={args.mask_strategy}, merge_method={args.dare_merge_method}")
        if args.dare_merge_method == "task_arithmetic":
            print(f"  λ = {lambdas}")
        else:
            print(f"  ties: λ={args.lamda}, density={args.density}, "
                  f"sign={args.sign_method}, func={args.merge_func}")
        final_sd = run_dare(
            base_sd, expert_sds, lambdas,
            weight_mask_rate=args.weight_mask_rate,
            use_rescale=args.use_rescale,
            mask_strategy=args.mask_strategy,
            merge_method=args.dare_merge_method,
            ties_lamda=args.lamda,
            ties_density=args.density,
            ties_sign_method=args.sign_method,
            ties_merge_func=args.merge_func,
            device=args.device,
        )

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "star":
        # Paper default: uniform average (1/N). Custom weights via --lambdas
        lambdas = args.lambdas or [1.0 / N] * N
        if len(lambdas) != N:
            parser.error(f"--lambdas count ({len(lambdas)}) does not match number of models ({N}).")
        print(f"  eta={args.eta}, λ={lambdas}")
        final_sd = run_star(base_sd, expert_sds, args.eta, lambdas, args.device)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "cart":
        print(f"  prior={args.prior}, rank_ratio={args.rank_ratio}")
        final_sd = run_cart(base_sd, expert_sds, args.prior, args.rank_ratio, args.device)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "tsv":
        if args.k is not None and args.sv_reduction is not None:
            parser.error("--k and --sv_reduction cannot be used simultaneously.")
        print(f"  α={args.alpha}, k={args.k}, sv_reduction={args.sv_reduction}")
        final_sd = run_tsv(base_sd, expert_sds, args.alpha, args.k, args.sv_reduction, args.device)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "fisher":
        lambdas = args.lambdas or [1.0] * N
        if len(lambdas) != N:
            parser.error(f"--lambdas count ({len(lambdas)}) does not match number of models ({N}).")
        final_sd = run_fisher(
            base_sd, expert_sds, args.expert_models,
            lambdas, args.epsilon,
            args.fisher_files, args.calib_samples, args.calib_seqlen,
            args.fisher_device, args.save_fisher, args.save_dir,
        )

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "iso_c":
        print(f"  alpha={args.alpha}")
        final_sd = run_iso_c(base_sd, expert_sds, args.alpha, args.device)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "iso_cts":
        print(f"  alpha={args.alpha}, common_space_fraction={args.common_space_fraction}")
        final_sd = run_iso_cts(
            base_sd, expert_sds, args.alpha, args.common_space_fraction, args.device,
        )

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "ram":
        print(f"  threshold={args.ram_threshold}, device={args.device}")
        final_sd = run_ram(base_sd, expert_sds, threshold=args.ram_threshold, device=args.device)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "ram_plus":
        print(f"  threshold={args.ram_threshold}, rescale_factor={args.rescale_factor}, device={args.device}")
        final_sd = run_ram_plus(
            base_sd, expert_sds,
            threshold=args.ram_threshold,
            rescale_factor=args.rescale_factor,
            device=args.device,
        )

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    del base_sd
    gc.collect()

    print(f"\nDone! Output: {args.save_dir}/")


if __name__ == "__main__":
    main()
