from enum import Enum

import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "mechanistic_interpretability"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_CHECKPOINTS_DIR = MODEL_DIR / "vision/checkpoints"
SAE_CHECKPOINTS_DIR = MODEL_DIR / "sae"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Evaluation(Enum):
    """Enum that maps to the possible evaluation functions"""

    FEATURE_BASIS_EVAL = "evaluate_sae_features"
    NEURON_BASIS_EVAL = "find_top_activations_for_neurons"


class EvaluationContext(Enum):

    TRAINING = "train"
    POST_TRAINING = "post-train"
