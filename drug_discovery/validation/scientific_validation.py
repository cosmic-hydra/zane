"""
Scientific Validation Framework for ZANE.

Rigorous statistical evaluation for drug discovery models:
- Scaffold-aware k-fold cross-validation
- Paired statistical significance tests (Wilcoxon, paired t-test)
- Bootstrap confidence intervals
- Comprehensive metrics: RMSE, MAE, R2, Pearson, Spearman, AUROC, AUPRC, EF
- Reproducibility utilities (seed management, config hashing)
- Structured experiment reports
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    logger.info(f"Global seed set to {seed}")


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


# --- Metrics ---
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / max(ss_tot, 1e-12))


def pearson_r(y_true, y_pred):
    if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_rho(y_true, y_pred):
    from scipy.stats import spearmanr

    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else 0.0


def auroc(y_true, y_score):
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def auprc(y_true, y_score):
    from sklearn.metrics import average_precision_score

    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return 0.0


def enrichment_factor(y_true, y_score, fraction=0.01):
    n = len(y_true)
    top_k = max(1, int(n * fraction))
    top_idx = np.argsort(-y_score)[:top_k]
    n_act = y_true.sum()
    return float((y_true[top_idx].sum() / top_k) / (n_act / n)) if n_act > 0 else 0.0


REGRESSION_METRICS = {"rmse": rmse, "mae": mae, "r2": r_squared, "pearson_r": pearson_r, "spearman_rho": spearman_rho}
CLASSIFICATION_METRICS = {"auroc": auroc, "auprc": auprc, "enrichment_1pct": lambda y, s: enrichment_factor(y, s, 0.01)}


def compute_metrics(y_true, y_pred, task_type="regression"):
    suite = REGRESSION_METRICS if task_type == "regression" else CLASSIFICATION_METRICS
    return {name: fn(y_true, y_pred) for name, fn in suite.items()}


# --- Scaffold splits ---
def bemis_murcko_scaffold(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(smiles)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smiles
    except ImportError:
        return smiles


def scaffold_split(smiles_list, frac_train=0.8, frac_val=0.1, seed=42):
    scaffolds = defaultdict(list)
    for i, s in enumerate(smiles_list):
        scaffolds[bemis_murcko_scaffold(s)].append(i)
    sets = sorted(scaffolds.values(), key=len, reverse=True)
    rng = random.Random(seed)
    rng.shuffle(sets)
    n = len(smiles_list)
    tc = int(n * frac_train)
    vc = int(n * (frac_train + frac_val))
    train, val, test = [], [], []
    for ss in sets:
        if len(train) + len(ss) <= tc:
            train.extend(ss)
        elif len(train) + len(val) + len(ss) <= vc:
            val.extend(ss)
        else:
            test.extend(ss)
    return train, val, test


def scaffold_kfold(smiles_list, n_folds=5, seed=42):
    scaffolds = defaultdict(list)
    for i, s in enumerate(smiles_list):
        scaffolds[bemis_murcko_scaffold(s)].append(i)
    sets = list(scaffolds.values())
    rng = random.Random(seed)
    rng.shuffle(sets)
    folds = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for ss in sorted(sets, key=len, reverse=True):
        smallest = min(range(n_folds), key=lambda i: sizes[i])
        folds[smallest].extend(ss)
        sizes[smallest] += len(ss)
    return [(sum([folds[j] for j in range(n_folds) if j != i], []), folds[i]) for i in range(n_folds)]


# --- Statistical tests ---
def paired_ttest(a, b):
    from scipy.stats import ttest_rel

    s, p = ttest_rel(a, b)
    return {"t_statistic": float(s), "p_value": float(p), "significant_005": p < 0.05}


def wilcoxon_test(a, b):
    from scipy.stats import wilcoxon

    try:
        s, p = wilcoxon(a, b)
        return {"statistic": float(s), "p_value": float(p), "significant_005": p < 0.05}
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0, "significant_005": False}


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    boot = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "ci_lower": float(np.percentile(boot, 100 * alpha)),
        "ci_upper": float(np.percentile(boot, 100 * (1 - alpha))),
    }


# --- Experiment Report ---
@dataclass
class ExperimentReport:
    experiment_id: str = ""
    config_hash: str = ""
    seed: int = 42
    model_name: str = ""
    dataset: str = ""
    split_method: str = "scaffold"
    n_folds: int = 5
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    aggregate_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    significance_tests: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    duration_seconds: float = 0.0

    def compute_aggregates(self):
        if not self.fold_metrics:
            return
        for name in self.fold_metrics[0]:
            vals = np.array([f[name] for f in self.fold_metrics])
            self.aggregate_metrics[name] = bootstrap_ci(vals)

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Report saved: {path}")
