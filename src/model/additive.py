import sys
import logging
import numpy as np
import torch
from pathlib import Path

# Add scGPT to path (reusing logic from wrapper.py)
current_dir = Path(__file__).parent.parent.parent
scgpt_path = current_dir / "scGPT"

if not scgpt_path.exists():
    scgpt_path = Path.cwd() / "scGPT"

if str(scgpt_path) not in sys.path:
    sys.path.insert(0, str(scgpt_path))

try:
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
except ImportError:
    logging.warning("Could not import scgpt directly. Checking path...")
    if not scgpt_path.exists():
        raise ImportError(f"scGPT directory not found at {scgpt_path}")
    else:
        raise


class AdditiveBaselineWrapper:
    """
    Additive Baseline (加和基线).

    For a double perturbation (A+B), predict:
        y_hat = y_A + y_B - y_ctrl

    where y_A and y_B are mean observed (log) expression vectors of the
    corresponding single perturbations from *training* data, and y_ctrl is
    the mean (log) expression vector of control samples.

    This baseline **does not use any double-perturbation training samples**.
    """

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.vocab = None

        # Stored statistics (numpy arrays, shape: (n_genes,))
        self.control_mean = None
        self.single_means = {}  # dict[str, np.ndarray]

        self._is_fitted = False
        self._load_vocab()

    def _load_vocab(self):
        # We need the vocab to ensure data loader compatibility
        model_dir = Path(self.config["paths"]["model_dir"])
        vocab_file = model_dir / "vocab.json"

        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        self.vocab = GeneVocab.from_file(vocab_file)

        # Ensure special tokens (matching ScGPTWrapper logic)
        special_tokens = [self.config["model"]["pad_token"], "<cls>", "<eoc>"]
        for s in special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)

        self.logger.info(f"Loaded vocab from {vocab_file}")

    @staticmethod
    def _is_double_condition(cond: str) -> bool:
        # Norman-style double perturbations are usually "A+B"
        return ("+" in cond) or ("&" in cond) or ("|" in cond) or ("," in cond)

    @staticmethod
    def _split_condition(cond: str):
        # Return list of perturbation names in the condition
        # Prefer '+' split; fall back to other common delimiters.
        for sep in ["+", "&", "|", ","]:
            if sep in cond:
                parts = [p.strip() for p in cond.split(sep) if p.strip()]
                return parts
        return [cond.strip()]

    def fit_from_adata(self, adata, condition_key="condition", control_key="ctrl"):
        """
        Fit additive baseline stats from AnnData.

        Parameters
        ----------
        adata:
            AnnData with expression in .X (log expression values).
        condition_key:
            Column in adata.obs indicating perturbation condition.
        control_key:
            Value of adata.obs[condition_key] indicating control (no perturbation).
        """
        self.logger.info("Fitting additive baseline from AnnData...")

        if condition_key not in adata.obs:
            raise KeyError(
                f"condition_key '{condition_key}' not found in adata.obs. "
                f"Available keys: {list(adata.obs.columns)[:20]}"
            )

        # Helper to get expression matrix as dense numpy
        def _to_dense(X):
            if hasattr(X, "toarray"):
                return X.toarray()
            return np.asarray(X)

        # Control mean
        ctrl_mask = adata.obs[condition_key] == control_key
        if ctrl_mask.sum() == 0:
            raise ValueError(
                f"No control samples found where obs['{condition_key}'] == '{control_key}'."
            )
        ctrl_X = _to_dense(adata[ctrl_mask].X)
        self.control_mean = np.mean(ctrl_X, axis=0)

        # Single perturbation means (exclude control + exclude double conditions)
        single_mask = (adata.obs[condition_key] != control_key)
        conds = adata.obs.loc[single_mask, condition_key].astype(str).values

        unique_conds = sorted(set(conds.tolist()))
        n_singles = 0
        for cond in unique_conds:
            if self._is_double_condition(cond):
                continue
            cond_mask = adata.obs[condition_key].astype(str) == str(cond)
            cond_X = _to_dense(adata[cond_mask].X)
            if cond_X.shape[0] == 0:
                continue
            self.single_means[str(cond)] = np.mean(cond_X, axis=0)
            n_singles += 1

        self._is_fitted = True
        self.logger.info(
            f"Additive baseline fitted: "
            f"control_mean from {ctrl_X.shape[0]} control samples; "
            f"{n_singles} single perturbations cached; "
            f"{self.control_mean.shape[0]} genes"
        )

    def predict(self, batch_data, gene_ids, target_gene=None, include_zero_gene="batch-wise", amp=True):
        """
        Predict expression for a batch, using target_gene to select (A,B).

        Notes:
        - batch_data is only used for batch size / device compatibility.
        - gene_ids/include_zero_gene/amp exist for API compatibility.

        Parameters
        ----------
        target_gene:
            Perturbation identifier (e.g. "A+B"). If None, will raise.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Additive baseline has not been fitted. "
                "Call fit_from_adata() with training data first."
            )
        if target_gene is None:
            raise ValueError("Additive baseline requires 'target_gene' (e.g., 'A+B') for prediction.")

        batch_size = batch_data.x.shape[0]
        n_genes = batch_data.x.shape[2]

        parts = self._split_condition(str(target_gene))
        if len(parts) == 1:
            a = parts[0]
            y_a = self.single_means.get(a, None)
            if y_a is None:
                self.logger.warning(f"[AdditiveBaseline] Missing single mean for '{a}'. Using control_mean (no effect).")
                y_hat = self.control_mean.copy()
            else:
                y_hat = y_a
        else:
            a, b = parts[0], parts[1]
            y_a = self.single_means.get(a, None)
            y_b = self.single_means.get(b, None)

            # If missing, fall back to control_mean (effect 0) for that component
            if y_a is None:
                self.logger.warning(f"[AdditiveBaseline] Missing single mean for '{a}'. Using control_mean (no effect).")
                y_a = self.control_mean
            if y_b is None:
                self.logger.warning(f"[AdditiveBaseline] Missing single mean for '{b}'. Using control_mean (no effect).")
                y_b = self.control_mean

            y_hat = (y_a + y_b - self.control_mean)

        # Guard: gene dimension mismatch
        if y_hat.shape[0] != n_genes:
            self.logger.warning(
                f"Gene count mismatch: cached mean has {y_hat.shape[0]} genes, input has {n_genes} genes. "
                "Truncating/padding cached mean to match input."
            )
            if y_hat.shape[0] > n_genes:
                y_hat = y_hat[:n_genes]
            else:
                y_hat = np.concatenate([y_hat, np.zeros(n_genes - y_hat.shape[0])], axis=0)

        pred = torch.from_numpy(np.asarray(y_hat)).float().unsqueeze(0).expand(batch_size, -1)

        # Move to same device as input
        if batch_data.x.is_cuda:
            pred = pred.cuda()

        return pred
