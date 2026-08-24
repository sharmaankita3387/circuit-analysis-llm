"""
Description:
    Provides utility routines to extract hidden-layer activations using TransformerLens
    and fit/evaluate linear logistic regression probes across model residual streams.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from transformer_lens import HookedTransformer


def extract_residual_activations(
    model: HookedTransformer,
    prompts: List[str],
    token_position: int = -1,
) -> Dict[int, np.ndarray]:
    """
    Extracts residual stream activations at a specified token position for all model layers.

    Args:
        model: Loaded HookedTransformer instance.
        prompts: List of input prompt strings.
        token_position: Index of token from which to extract activations (-1 indicates final token).

    Returns:
        Dict mapping layer index (int) to a 2D numpy array of shape (len(prompts), d_model).
    """
    n_layers = model.cfg.n_layers
    all_layer_acts: Dict[int, List[np.ndarray]] = {l: [] for l in range(n_layers)}

    with torch.no_grad():
        for p in prompts:
            _, cache = model.run_with_cache(p)
            for l in range(n_layers):
                resid_act = (
                    cache[f"blocks.{l}.hook_resid_post"][0, token_position, :]
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )
                all_layer_acts[l].append(resid_act)

    return {l: np.array(acts) for l, acts in all_layer_acts.items()}


def train_and_eval_layer_probes(
    layer_acts: Dict[int, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, float]], Dict[int, LogisticRegression]]:
    """
    Trains linear Logistic Regression probes on each layer's activations and evaluates
    classification accuracy, ROC-AUC, and False Positive Rate (FPR) on a test partition.

    Args:
        layer_acts: Dict of layer indices to numpy activation arrays.
        y_train: Training split binary labels.
        y_test: Test split binary labels.
        train_idx: Integer indices corresponding to training examples.
        test_idx: Integer indices corresponding to testing examples.
        random_seed: Seed for solver determinism.

    Returns:
        A tuple of (metrics_list, trained_probes_dict).
    """
    metrics = []
    trained_probes = {}

    for layer, acts in layer_acts.items():
        X_train, X_test = acts[train_idx], acts[test_idx]

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=random_seed)
        clf.fit(X_train, y_train)
        trained_probes[layer] = clf

        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        metrics.append({
            "layer": layer,
            "accuracy": float(acc),
            "roc_auc": float(auc),
            "fpr": float(fpr),
        })

    return metrics, trained_probes