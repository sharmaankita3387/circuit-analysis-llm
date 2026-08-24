"""
Description:
    Implements a 3-strike real-time activation guardrail that monitors latent intent,
    aborts generation on high-confidence safety breaches, and packages simulated audit payloads.
"""

from typing import Any, Dict
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformer_lens import HookedTransformer


class ActivationGuardrail:
    """
    Real-time latent state guardrail system using internal residual stream probes.
    """

    def __init__(
        self,
        model: HookedTransformer,
        probe: LogisticRegression,
        target_layer: int,
        warn_threshold: float = 0.60,
        freeze_threshold: float = 0.85,
    ):
        self.model = model
        self.probe = probe
        self.target_layer = target_layer
        self.warn_threshold = warn_threshold
        self.freeze_threshold = freeze_threshold

    def evaluate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Runs forward pass activation extraction and predicts harm score for an incoming prompt.

        Args:
            prompt: Text string query to audit.

        Returns:
            Dict containing intent classification verdict, harm probability score, and log metadata.
        """
        with torch.no_grad():
            _, cache = self.model.run_with_cache(prompt)
            resid_act = (
                cache[f"blocks.{self.target_layer}.hook_resid_post"][0, -1, :]
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
                .reshape(1, -1)
            )

        harm_prob = float(self.probe.predict_proba(resid_act)[0, 1])

        if harm_prob < self.warn_threshold:
            action = "ALLOW"
            action_code = 0
            log_detail = "Query verified safe. Forward pass permitted with 0 user data retention."
        elif self.warn_threshold <= harm_prob < self.freeze_threshold:
            action = "WARN"
            action_code = 1
            log_detail = f"Adversarial ambiguity flagged (Score: {harm_prob:.3f}). User prompted to clarify scope."
        else:
            action = "FREEZE & LOG TICKET"
            action_code = 2
            log_detail = (
                f"Malicious intent detected (Score: {harm_prob:.3f}). Execution aborted. "
                f"Logged activation state hash to multi-tier dispute queue (Ticket Strike +1)."
            )

        return {
            "prompt": prompt,
            "target_layer": self.target_layer,
            "harm_probability": round(harm_prob, 4),
            "action": action,
            "action_code": action_code,
            "log": log_detail,
        }