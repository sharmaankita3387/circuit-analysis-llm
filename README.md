# Activation-Guided Intent Auditing & Dynamic Escalation Guardrails

This repository implements internal representation monitoring across transformer residual streams to identify adversarial user intent and evaluate dynamic multi-tiered guardrail escalation policies.

## 1. Project Overview
- **Core Hypothesis:** Latent representations in intermediate residual stream layers exhibit strong linear separation between malicious/policy-violating intent and benign ambiguity before text generation completes.
- **Threat Model:** Adversarial jailbreaks, indirect injections, and illicit tooling requests targeting autonomous personal assistants.
- **Privacy Guarantee:** Zero persistent user conversational logging during standard interaction ($S < 0.60$), with state-activation hash recording strictly reserved for high-confidence policy breaches ($S \ge 0.85$).

## 2. Repository Architecture
```text
circuit-analysis-llm/
├── notebooks/
│   └── main_investigation.ipynb   # Complete probing pipeline & visualizations
├── results/
│   └── layer_probe_metrics.png    # Layer-wise probe performance chart
├── src/
│   ├── probing.py                 # Residual stream extraction & linear probe training
│   └── guardrail.py               # Real-time inference monitor & escalation policy
├── requirements.txt
└── README.md