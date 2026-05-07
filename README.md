# Policy-Shift-Guided Spectral Alignment (PSA)

> **Pre-release.** Reference implementation for the preprint
> *Merging RLVR-Trained Experts via Policy-Shift-Guided Spectral Alignment*.
> Paper / project page links will be added on release.

PSA is a retraining-free method for merging multiple RLVR post-trained
language-model experts into one model. Existing SFT-oriented merging recipes
operate on global parameter geometry and under-preserve the sparse, directional
next-token-probability changes that distinguish RLVR experts from the base.
PSA scores calibration tokens by expert–base probability shift, converts
shift-weighted input activations into column weights for a weighted-SVD
subspace selection, and then polar-aligns the retained bases across experts
and rescales each per-layer Frobenius norm before aggregation.

Two variants ship out of the box:

| variant      | base                       | 3 RLVR experts                |
|--------------|----------------------------|-------------------------------|
| Qwen2.5-7B   | `Qwen/Qwen2.5-7B-Instruct` | coding · tool · memory        |
| Qwen3-1.7B   | `Qwen/Qwen3-1.7B-Base`     | if · math · long-context      |

## Repository structure

```
PSA/
├── pipeline/    PSA reference implementation (3-step merge)
├── baselines/   9 merging baselines (TA, TIES, DARE, TSV, ...)
├── eval/        6-benchmark eval harness (AIME · LCB · IFEval · LiveBench · BFCL · RULER)
└── docs/        project page (GitHub Pages)
```

See each subdirectory's `README.md` for details.

## Quick start

```bash
pip install torch numpy tqdm transformers safetensors huggingface_hub

# Qwen2.5-7B variant — proxy data bundled, ~50 min on 2× A6000
bash pipeline/run_qwen2.5_7b.sh

# Qwen3-1.7B variant — auto-generates proxy on first run, ~25 min
bash pipeline/run_qwen3_1.7b.sh
```

The merged checkpoint is written as HF-format safetensors. Run evaluation
against the 6-benchmark suite from `eval/`:

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
