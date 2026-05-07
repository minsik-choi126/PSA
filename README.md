# Policy-Shift-Guided Spectral Alignment (PSA)

> **Pre-release.** Reference implementation for the preprint
> *Merging RLVR-Trained Experts via Policy-Shift-Guided Spectral Alignment*.
> Paper / project page links will be added on release.

PSA is a retraining-free, three-stage method for merging multiple RLVR
post-trained language-model experts into one model.
Existing SFT-oriented merging recipes — coordinate-wise (TA / TIES / DARE) and
spectral (TSV / STAR / Iso-CTS) — select what to keep using *global* parameter
geometry and under-preserve the sparse, directional next-token-probability
shifts that distinguish RLVR experts from the base. PSA addresses this in
three stages: (1) score calibration tokens by absolute expert-vs-base
log-probability difference and convert shift-weighted input activations into
per-column importance weights; (2) perform a *column-metric* low-rank
truncation that prioritizes directions active on high-shift tokens; (3)
polar-align the retained bases across experts and Frobenius-renormalize each
expert's update before aggregation.

Two variants ship out of the box:

| variant      | base                       | 3 RLVR experts                                  |
|--------------|----------------------------|-------------------------------------------------|
| Qwen2.5-7B   | `Qwen/Qwen2.5-7B-Instruct` | CURE (coding) · ToolRL (tool) · MemAgent (memory) |
| Qwen3-1.7B   | `Qwen/Qwen3-1.7B-Base`     | IF (custom RL) · math (custom RL) · Lucy (search) |

## Repository structure

```
PSA/
├── pipeline/    PSA reference implementation (3-stage merge)
├── baselines/   8 merging baselines (TA · TIES · DARE · TSV · STAR · Iso-CTS · RAM · RAM+)
├── eval/        evaluation harness — LiveBench · LiveCodeBench · BFCL · RULER · IFEval · AIME · SimpleQA · BrowseComp
└── docs/        project page (GitHub Pages)
```

See each subdirectory's `README.md` for details.

## Environment

PSA targets Python ≥ 3.10 and a CUDA-enabled PyTorch ≥ 2.1. Pick the install
that matches your scope:

```bash
# Just the merge pipeline (extract_w → apply_wnorm → merge):
pip install -r pipeline/requirements.txt

# Evaluation harness (vLLM, flash-attn, VERL installed by eval/setup.sh):
bash eval/setup.sh

# Everything in one shot:
pip install -r requirements.txt
```

Heavy deps (vLLM, flash-attn, VERL, BFCL) have their own install paths and are
handled by `eval/setup.sh` — see [`eval/README.md`](eval/README.md).

## Quick start

```bash
# Qwen2.5-7B variant — proxy data bundled, ~50 min on 2× A6000
bash pipeline/run_qwen2.5_7b.sh

# Qwen3-1.7B variant — auto-generates proxy on first run, ~25 min
bash pipeline/run_qwen3_1.7b.sh
```

The merged checkpoint is written as HF-format safetensors. Then run evaluation:

```bash
bash eval/setup.sh
bash eval/run_eval.sh --model ./merged_qwen2.5_7b --benchmarks all
```

## Citation

```bibtex
@misc{psa2026,
  title  = {Merging RLVR-Trained Experts via Policy-Shift-Guided Spectral Alignment},
  author = {(authors)},
  year   = {2026},
  note   = {Preprint},
}
```

## License

Code under `pipeline/`, `baselines/`, `eval/` is released under the MIT License
(see `LICENSE`). Project page assets under `docs/static/` are adapted from the
[Nerfies](https://github.com/nerfies/nerfies.github.io) template under
CC BY-SA 4.0.
