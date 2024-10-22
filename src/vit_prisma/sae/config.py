import math
import os
from abc import ABC
from dataclasses import dataclass
from dataclasses import fields, field
from typing import Any, Optional, Dict, List

import torch
from transformers import ViTConfig

from vit_prisma.utils.constants import Evaluation


@dataclass
class RunnerConfig(ABC):
    """
    The config that's shared across all runners.
    """

    # Data Generating Function (Model + Training Distibuion)
    model_class_name: str = "HookedViT"
    model_name: str = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"
    vit_model_cfg: Optional[ViTConfig] = None
    model_path: str = None
    hook_point_layer: int = 9
    hook_point_head_index: Optional[int] = None
    context_size: int = 50
    use_cached_activations: bool = False
    cached_activations_path: Optional[str] = (
        None  # Defaults to "activations/{dataset}/{model}/{full_hook_name}_{hook_point_head_index}"
    )

    # SAE Parameters
    d_in: int = 512
    activation_fn_str: str = "relu" # relu or topk
    activation_fn_kwargs: dict[str, Any] = field(default_factory=dict)

    # SAE Training run tolerance
    min_l0 = None
    min_explained_variance = None

    # New changes
    max_grad_norm: float = 1.0 # For gradient clipping, set to None to turn off
    initialization_method: str = "encoder_transpose_decoder" # or independent
    normalize_activations: str = "layer_norm"

    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    store_batch_size: int = 32

    # Training length parameters
    num_epochs: int = 1
    total_training_images: int = int(1_300_000*num_epochs) # To do: make this not hardcoded
    total_training_tokens: int = total_training_images * context_size # Images x tokens

    image_size: int = 224

    # Misc
    device: str | torch.device = "cpu"
    seed: int = 42
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.hook_point = f"blocks.{self.hook_point_layer}.hook_mlp_out" # change hookpoint name here

        self.num_patch = int(math.sqrt(self.context_size - 1))

        # Autofill cached_activations_path unless the user overrode it
        if self.cached_activations_path is None:
            self.cached_activations_path = f"activations/{self.dataset_path.replace('/', '_')}/{self.model_name.replace('/', '_')}/{self.hook_point}"
            if self.hook_point_head_index is not None:
                self.cached_activations_path += f"_{self.hook_point_head_index}"

    def pretty_print(self):
        print("Configuration:")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.dtype):
                value = str(value).split('.')[-1]  # Convert torch.dtype to string
            elif isinstance(value, torch.device):
                value = str(value)  # Convert torch.device to string
            print(f"  {field.name}: {value}")



@dataclass
class EvalConfig(ABC):
    evaluation_functions: List[Evaluation]
    eval_frequency: Optional[int]
    batch_size: int
    max_evaluation_images: int
    samples_per_bin: int  # Number of features to sample per pre-specified interval
    max_images_per_feature: int  # Number of max images to collect per feature


@dataclass
class TrainingEvalConfig(EvalConfig):
    """Default values to be used when evaluating an SAE during training. These are less
    compute and time costly as they get run every step.
    """

    evaluation_functions: List[Evaluation] = field(
        default_factory=lambda: []
    )
    eval_frequency: Optional[int] = 2500
    batch_size: int = 28
    max_evaluation_images: int = 100
    samples_per_bin: int = 10
    max_images_per_feature: int = 20


@dataclass
class PostTrainingEvalConfig(EvalConfig):
    """Default values to be used when evaluating an SAE post-training. These evaluations
    are more extensive as they are run to get a complete picture of the quality of the
    SAE.
    """

    evaluation_functions: List[Evaluation] = field(
        default_factory=lambda: [
            Evaluation.FEATURE_BASIS_EVAL,
            # Evaluation.NEURON_BASIS_EVAL,
        ]
    )
    eval_frequency: Optional[int] = 1
    batch_size: int = 28
    max_evaluation_images: int = 100
    samples_per_bin: int = 10
    max_images_per_feature: int = 20


@dataclass
class VisionModelSAERunnerConfig(RunnerConfig):
    """
    Configuration for training a sparse autoencoder on a vision model.
    """

    # Logging
    verbose: bool = False

    # SAE Parameters
    b_dec_init_method: str = "geometric_median"
    expansion_factor: int = 16
    from_pretrained_path: Optional[str] = None
    d_sae: Optional[int] = None

    # Training Parameters
    l1_coefficient: float = 0.0002 # 0.00008
    lp_norm: float = 1
    lr: float = 0.001
    lr_scheduler_name: str = (
        "cosineannealingwarmup"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup
    )
    lr_warm_up_steps: int = 500

    train_batch_size: int = 1024*4

    # Imagenet1k
    dataset_name: str = 'imagenet1k'  # imagenet1k | cifar10
    dataset_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets"
    dataset_train_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train"
    dataset_val_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val"
   
    # Resampling protocol args
    use_ghost_grads: bool = True
    feature_sampling_window: int = 1000 # 1000
    dead_feature_window: int = 5000  # unless this window is larger feature sampling,

    dead_feature_threshold: float = 1e-8

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "tinyclip_sae_16_hyperparam_sweep_lr"
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = 100

    # Misc
    n_checkpoints: int = 10
    checkpoint_path: str = "/network/scratch/s/sonia.joseph/sae_checkpoints/tinyclip_40M_mlp_out"

    # Evaluation
    sae_path: str = '/network/scratch/s/sonia.joseph/sae_checkpoints/tinyclip_40M_mlp_out/1f89d99e-wkcn-TinyCLIP-ViT-40M-32-Text-19M-LAION400M-expansion-16/n_images_520028.pt'
    training_eval: EvalConfig = TrainingEvalConfig()
    post_training_eval: EvalConfig = PostTrainingEvalConfig()

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.expansion_factor, list):
            self.d_sae = self.d_in * self.expansion_factor
        self.tokens_per_buffer = (
            self.train_batch_size * self.context_size * self.n_batches_in_buffer
        )
        if self.b_dec_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.b_dec_init_method}"
            )
        if self.b_dec_init_method == "zeros":
            print(
                "Warning: We are initializing b_dec to zeros. This is probably not what you want."
            )

        self.device = torch.device(self.device)

        # Print out some useful info:
        n_tokens_per_buffer = (
            self.store_batch_size * self.context_size * self.n_batches_in_buffer
        )
        print(f"n_tokens_per_buffer (millions): {n_tokens_per_buffer / 10 **6}")
        n_contexts_per_buffer = self.store_batch_size * self.n_batches_in_buffer
        print(
            f"Lower bound: n_contexts_per_buffer (millions): {n_contexts_per_buffer / 10 **6}"
        )

        self.total_training_steps = self.total_training_tokens // self.train_batch_size
        print(f"Total training steps: {self.total_training_steps}")

        print(f"Total training images: {self.total_training_images}")

        total_wandb_updates = self.total_training_steps // self.wandb_log_frequency
        print(f"Total wandb updates: {total_wandb_updates}")

        # print expansion factor
        print(f"Expansion factor: {self.expansion_factor}")

        # how many times will we sample dead neurons?
        # assert self.dead_feature_window <= self.feature_sampling_window, "dead_feature_window must be smaller than feature_sampling_window"
        n_feature_window_samples = self.total_training_steps // self.feature_sampling_window
        print(
            f"n_tokens_per_feature_sampling_window (millions): {(self.feature_sampling_window * self.context_size * self.train_batch_size) / 10 **6}"
        )
        print(
            f"n_tokens_per_dead_feature_window (millions): {(self.dead_feature_window * self.context_size * self.train_batch_size) / 10 **6}"
        )

        if self.use_ghost_grads:
            print("Using Ghost Grads.")

        print(
            f"We will reset the sparsity calculation {n_feature_window_samples} times."
        )
        # print("Number tokens in dead feature calculation window: ", self.dead_feature_window * self.train_batch_size)
        print(
            f"Number tokens in sparsity calculation window: {self.feature_sampling_window * self.train_batch_size:.2e}")
        
        if self.max_grad_norm:
            print(f"Gradient clipping with max_norm={self.max_grad_norm}")
        
        # Print initialization method
        print(
            f"Using SAE initialization method: {self.initialization_method}"
        )

        self.activation_fn_kwargs = self.activation_fn_kwargs or {}


@dataclass
class CacheActivationsRunnerConfig(RunnerConfig):
    """Configuration for caching activations of an LLM.""";

    # Activation caching stuff
    shuffle_every_n_buffers: int = 10;
    n_shuffles_with_last_section: int = 10;
    n_shuffles_in_entire_dir: int = 10;
    n_shuffles_final: int = 100;

    def __post_init__(self):
        super().__post_init__();
        if self.use_cached_activations:
            raise ValueError("Use_cached_activations should be False when running cache_activations_runner");
