# 文件: models/algorithm.py
"""Compatibility exports for the federated LoRA algorithm modules.

The implementation has been split into smaller modules:
- models.client: Client
- models.server: Server
- models.scheduling: Configuration and scheduling mixin
- models.metrics: evaluation helpers
"""

from models.client import Client
from models.server import Server
from models.scheduling import Configuration
from models.metrics import (
    evaluate_accuracy,
    evaluate_macro_f1,
    evaluate_model_loss,
    evaluate_rouge_l,
    normalize_text,
)

__all__ = [
    "Client",
    "Server",
    "Configuration",
    "evaluate_accuracy",
    "evaluate_macro_f1",
    "evaluate_model_loss",
    "evaluate_rouge_l",
    "normalize_text",
]
