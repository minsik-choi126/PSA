# PSA — Baselines

Self-contained implementations of the 8 merging baselines reported in the PSA
paper, plus a few additional methods supported in the same single-file driver.
Single-file (`merge_baseline.py`, ~1.4k lines) so each method can be inspected
and run independently of the PSA pipeline.

## Methods

Paper baselines (Table 1, Table 2):

| method            | `--method`        | summary                                              |
|-------------------|-------------------|------------------------------------------------------|
| Task Arithmetic   | `task_arithmetic` | mean of task vectors                                 |
| TIES              | `ties`            | sign-resolve + magnitude pruning + aggregate         |
| DARE              | `dare`            | drop-and-rescale + (TA \| TIES) aggregate            |
| TSV               | `tsv`             | per-expert truncated SVD + polar alignment           |
| STAR              | `star`            | spectral truncation + per-direction rescale          |
| Iso-CTS           | `iso_cts`         | common + task-specific subspaces with isotropic spec |
| RAM               | `ram`             | magnitude-mask shared/task-specific split + rescale  |
| RAM+              | `ram_plus`        | RAM with refined task-specific scaling               |

Additional methods supported in the same driver:

| method            | `--method`        | summary                                              |
|-------------------|-------------------|------------------------------------------------------|
| CART              | `cart`            | mean + residual decomposition of task vectors        |
| Fisher Merging    | `fisher`          | Fisher-information-weighted parameter averaging      |
| Iso-C             | `iso_c`           | Iso-CTS variant with common subspace only            |

## Usage

```bash
python merge_baseline.py \
    --base_model  <path-or-hf-id> \
    --experts     coding=<path>  tool=<path>  memory=<path> \
    --method      ties \
    --out_dir     ./merged_ties
```

The `--base_model` / `--experts` / `--out_dir` interface matches
`pipeline/merge.py`, so swapping methods only requires the `--method` flag.
See in-file docstrings for method-specific knobs (`--alpha`, `--ties_k`,
`--dare_drop_prob`, `--energy`, etc.).
