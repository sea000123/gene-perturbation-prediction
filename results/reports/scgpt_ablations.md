# Reverse Perturbation Prediction - Model Comparison

## Overview

Comparison of 3 experiments across 3 runs.

## Results Table

| experiment      | mask   |   exact_hit@1 |   exact_hit@5 |   relevant_hit@1 |   relevant_hit@5 |    mrr |   ndcg@5 |
|:----------------|:-------|--------------:|--------------:|-----------------:|-----------------:|-------:|---------:|
| scgpt           | True   |        0.0022 |        0.0194 |           0.071  |           0.2106 | 0.0091 |   0.0103 |
| scgpt           | False  |        0.0019 |        0.0107 |           0.0773 |           0.18   | 0.0086 |   0.0065 |
| scgpt_head_only | True   |        0.0019 |        0.0098 |           0.0716 |           0.1893 | 0.0078 |   0.006  |
| scgpt_head_only | False  |        0.0019 |        0.0213 |           0.0751 |           0.2234 | 0.0115 |   0.0117 |
| scgpt_lora_head | True   |        0.0022 |        0.0188 |           0.0781 |           0.2032 | 0.0102 |   0.0101 |
| scgpt_lora_head | False  |        0.0019 |        0.009  |           0.0691 |           0.2319 | 0.0068 |   0.0054 |

## Notes

- **mask=True**: Anti-cheat masking enabled (perturbed gene expression zeroed)
- **mask=False**: No masking (includes potential leakage signal)
- **experiment/encoder**: Uses logging.experiment_name when multiple runs share an encoder; otherwise shows encoder
- **exact_hit@K**: Fraction where true condition is in top-K predictions
- **relevant_hit@K**: Fraction where any top-K prediction shares a gene with ground truth
- **mrr**: Mean Reciprocal Rank
- **ndcg@K**: Normalized Discounted Cumulative Gain at K
