# src/eval/h5ad_eval.py
'''
export PYTHONHASHSEED=0
python src/main_new.py \
  --config src/configs/h5ad.yaml \
  --eval_mode h5ad \
  --train_h5ad ../CellPerturb_VCC_Proj6/vcc_data/log1p/train_lognorm.h5ad \
  --truth_h5ad ../CellPerturb_VCC_Proj6/vcc_data/log1p/test_lognorm.h5ad \
  --pred_h5ad ../CellPerturb_VCC_Proj6/vcc_data/log1p/test_pred.h5ad \
  --pert_col target_gene \
  --control_label non-targeting 
'''
from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc

from src.utils.de_metrics import (
    compute_de_comparison_metrics,
    compute_mae_top2k,
    compute_pds,
    compute_pseudobulk_delta,
    compute_overall_score,
)

def _to_dense(X):
    return X if isinstance(X, np.ndarray) else X.toarray()

def evaluate_from_h5ad(
    train_h5ad: str,
    truth_h5ad: str,
    pred_h5ad: str,
    pert_col: str = "target_gene",
    control_label: str = "non-targeting",
    top_k: int = 2000,
    threads: int = -1,
    eval_batch_size: int = 16,
):
    """
    Strictly align evaluation to Code1 behavior:
      - control_mean computed from TRAIN control cells
      - control_expr sampled per-batch with replace=True using batch_seed derived from hash(target)
      - control_expr for the whole target is concatenated from batch samples
    """

    train = sc.read_h5ad(train_h5ad)
    truth = sc.read_h5ad(truth_h5ad)
    pred = sc.read_h5ad(pred_h5ad)

    # 1) Align genes across all three by var_names intersection
    common_genes = train.var_names.intersection(truth.var_names).intersection(pred.var_names)
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between train/truth/pred.")

    train = train[:, common_genes].copy()
    truth = truth[:, common_genes].copy()
    pred  = pred[:, common_genes].copy()

    gene_names = np.array(common_genes)

    # 2) Control cells come from TRAIN (Code1 uses train_adata control_cells)
    if pert_col not in train.obs:
        raise ValueError(f"Missing obs column {pert_col} in train.")
    train_ctrl = train[train.obs[pert_col].astype(str) == control_label]
    if train_ctrl.n_obs == 0:
        raise ValueError(f"No control cells found in train with label={control_label}.")

    Xc_train = _to_dense(train_ctrl.X)
    control_mean = Xc_train.mean(axis=0)  # Code1 get_control_mean()

    # 3) Targets come from truth (like Code1 data_loader.get_test_targets())
    if pert_col not in truth.obs or pert_col not in pred.obs:
        raise ValueError(f"Missing obs column {pert_col} in truth/pred.")

    targets = truth.obs[pert_col].astype(str).unique()

    results = []
    pred_deltas, truth_deltas, target_gene_indices = {}, {}, {}

    for target in targets:
        if target == control_label:
            continue

        truth_t = truth[truth.obs[pert_col].astype(str) == target]
        pred_t  = pred[pred.obs[pert_col].astype(str) == target]

        Xt = _to_dense(truth_t.X)
        Xp = _to_dense(pred_t.X)

        n_cells = Xt.shape[0]
        if n_cells == 0:
            continue

        # ---------- Code1-like per-batch control sampling ----------
        # base_seed derived from hash(target)
        base_seed = hash(target) % (2**32)

        control_expr_list = []
        i = 0
        for batch_start in range(0, n_cells, eval_batch_size):
            i += 1
            batch_end = min(batch_start + eval_batch_size, n_cells)
            batch_n = batch_end - batch_start

            # Code1 batch_seed
            batch_seed = base_seed + batch_start - (i % 5) * eval_batch_size

            rng = np.random.default_rng(batch_seed)
            idx = rng.choice(Xc_train.shape[0], batch_n, replace=True)
            batch_control_expr = Xc_train[idx]

            control_expr_list.append(batch_control_expr)

        control_expr = np.concatenate(control_expr_list, axis=0)

        # DES
        try:
            de = compute_de_comparison_metrics(
                control_expr=control_expr,
                pred_expr=Xp,
                truth_expr=Xt,
                gene_names=gene_names,
                fdr_threshold=0.05,
                threads=threads,
            )
        except Exception:
            de = {"des": np.nan, "n_de_truth": 0, "n_de_pred": 0, "n_intersect": 0}

        # MAE_top2k
        mae = compute_mae_top2k(
            pred_expr=Xp,
            truth_expr=Xt,
            control_mean=control_mean,
            top_k=top_k,
        )

        results.append({
            "perturbation_id": target,
            "n_cells": n_cells,
            "des_k": de["des"],
            "mae_top2000_k": mae,
            "n_true_de": de["n_de_truth"],
            "n_pred_de": de["n_de_pred"],
            "n_intersect": de.get("n_intersect", 0),
        })

        # PDS deltas (same as Code1)
        pred_deltas[target] = compute_pseudobulk_delta(Xp, control_mean)
        truth_deltas[target] = compute_pseudobulk_delta(Xt, control_mean)
        target_gene_indices[target] = -1

    results_df = pd.DataFrame(results)

    # 4) PDS + per-perturbation overall_score
    if len(pred_deltas) > 0:
        pds_result = compute_pds(pred_deltas, truth_deltas, target_gene_indices)
        pds_nrank = pds_result["npds"]
        pds_ranks = pds_result["ranks"]

        n_perts = len(pred_deltas)
        results_df["rank_Rk"] = results_df["perturbation_id"].map(pds_ranks)
        results_df["rank_Rk_norm"] = results_df["rank_Rk"] / n_perts

        def _row_score(r):
            out = compute_overall_score(r["rank_Rk_norm"], r["mae_top2000_k"], r["des_k"])
            return out["overall_score"]

        results_df["overall_score"] = results_df.apply(_row_score, axis=1)
    else:
        pds_nrank = np.nan
        results_df["rank_Rk"] = np.nan
        results_df["rank_Rk_norm"] = np.nan
        results_df["overall_score"] = np.nan

    # 5) Global Score of Averages (same as Code2)
    mean_des = results_df["des_k"].mean() if not results_df.empty else np.nan
    mean_mae = results_df["mae_top2000_k"].mean() if not results_df.empty else np.nan
    summary = compute_overall_score(pds_nrank, mean_mae, mean_des)
    summary.update({
        "pds_nrank": pds_nrank,
        "mean_des": mean_des,
        "mean_mae": mean_mae,
    })

    return results_df, summary
