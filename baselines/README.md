# PSA — Baselines

Self-contained reference implementations of 9 merging baselines used in the
PSA paper. Single-file (`merge_baseline.py`, ~1.4k lines) so each method can
be inspected and run independently of the PSA pipeline.

## Baselines covered

| method            | flag                | summary                                                |
|-------------------|---------------------|--------------------------------------------------------|
| Task Arithmetic   | `--method ta`       | mean of task vectors                                   |
| TIES              | `--method ties`     | sign-resolve + top-k pruning + mean                    |
| DARE              | `--method dare`     | drop-and-rescale random pruning                        |
| DARE-TA / DARE-TIES | `--method dare_ta` / `dare_ties` | DARE composed with TA / TIES        |
| TSV               | `--method tsv`      | task-vector spectral truncation + mean                 |
| ISO-CTS           | `--method iso_cts`  | isotropic random subspace                              |
| RAM / RAM+        | `--method ram` / `ram_plus` | rank-aware merging                              |
| STAR              | `--method star`     | spectral truncation with alignment                     |

## Usage

```bash
python merge_baseline.py \
    --base_model <path-or-hf-id> \
    --experts coding=<path> tool=<path> memory=<path> \
    --method ties --out_dir ./merged_ties
```

Each baseline accepts the same `--base_model` / `--experts` / `--out_dir`
interface as `pipeline/merge.py`, so swapping methods only requires the
`--method` flag. See in-file docstrings for method-specific knobs
(`--alpha`, `--top_k`, `--energy`, etc.).
