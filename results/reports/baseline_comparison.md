# Reverse Perturbation Prediction - Model Comparison

## Overview

Comparison of 2 models across 2 runs.

## Results Table

| encoder   | mask   |   exact_hit@1 |   exact_hit@5 |   relevant_hit@1 |   relevant_hit@5 |    mrr |   ndcg@5 |
|:----------|:-------|--------------:|--------------:|-----------------:|-----------------:|-------:|---------:|
| logreg    | True   |        0      |        0      |           0.4329 |           0.6233 | 0      |   0      |
| logreg    | False  |        0      |        0      |           0.4698 |           0.6394 | 0      |   0      |
| pca       | True   |        0.1759 |        0.4062 |           0.3903 |           0.587  | 0.2739 |   0.2956 |
| pca       | False  |        0.1819 |        0.4155 |           0.3958 |           0.5936 | 0.281  |   0.3035 |

## Notes

- **mask=True**: Anti-cheat masking enabled (perturbed gene expression zeroed)
- **mask=False**: No masking (includes potential leakage signal)
- **exact_hit@K**: Fraction where true condition is in top-K predictions
- **relevant_hit@K**: Fraction where any top-K prediction shares a gene with ground truth
- **mrr**: Mean Reciprocal Rank
- **ndcg@K**: Normalized Discounted Cumulative Gain at K
