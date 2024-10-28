"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import gzip
import os
import pickle
from typing import Any, Callable

import einops
import torch
from torch import nn


from vit_prisma.prisma_tools.hooked_root_module import HookedRootModule
from vit_prisma.prisma_tools.hook_point import HookPoint

from vit_prisma.sae.config import VisionModelSAERunnerConfig

# import fields
from dataclasses import fields


from vit_prisma.sae.training.geometric_median import compute_geometric_median # Note: this is the SAE Lens 3 version, not SAE Lens 2 version

import math


class SparseAutoencoder(HookedRootModule):
    """ """

    def __init__(
        self,
        cfg: VisionModelSAERunnerConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in}; {type(self.d_in)}"
            )
        assert cfg.d_sae is not None  # keep pyright happy
        self.d_sae = cfg.d_sae
        self.l1_coefficient = cfg.l1_coefficient
        self.lp_norm = cfg.lp_norm
        self.dtype = cfg.dtype
        self.device = cfg.device
        self.initialization_method = cfg.initialization_method


        
        # Initialize weights based on the chosen method
        if self.cfg.architecture == "standard":
            if self.initialization_method == "independent":
                self.W_dec = nn.Parameter(
                    self.initialize_weights(self.d_sae, self.d_in)
                )
                self.W_enc = nn.Parameter(
                    self.initialize_weights(self.d_in, self.d_sae)
                )
            elif self.initialization_method == "encoder_transpose_decoder":
                self.W_dec = nn.Parameter(
                    self.initialize_weights(self.d_sae, self.d_in)
                )
                self.W_enc = nn.Parameter(self.W_dec.data.t().clone())
            else:
                raise ValueError(f"Unknown initialization method: {self.initialization_method}")
        elif self.cfg.architecture == "gated":
            assert self.cfg.use_ghost_grads == False, "Gated SAE does not support ghost grads"
            self.initialize_weights_gated()

        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()

        self.zero_loss = None

        # handle run time activation normalization if needed:
        if self.cfg.normalize_activations == "constant_norm_rescale":

            #  we need to scale the norm of the input and store the scaling factor
            def run_time_activation_norm_fn_in(x: torch.Tensor) -> torch.Tensor:
                self.x_norm_coeff = (self.cfg.d_in**0.5) / x.norm(dim=-1, keepdim=True)
                x = x * self.x_norm_coeff
                return x

            def run_time_activation_norm_fn_out(x: torch.Tensor) -> torch.Tensor:  #
                x = x / self.x_norm_coeff
                del self.x_norm_coeff  # prevents reusing
                return x

            self.run_time_activation_norm_fn_in = run_time_activation_norm_fn_in
            self.run_time_activation_norm_fn_out = run_time_activation_norm_fn_out

        elif self.cfg.normalize_activations == "layer_norm":

            #  we need to scale the norm of the input and store the scaling factor
            def run_time_activation_ln_in(
                x: torch.Tensor, eps: float = 1e-5
            ) -> torch.Tensor:
                mu = x.mean(dim=-1, keepdim=True)
                x = x - mu
                std = x.std(dim=-1, keepdim=True)
                x = x / (std + eps)
                self.ln_mu = mu
                self.ln_std = std
                return x

            def run_time_activation_ln_out(x: torch.Tensor, eps: float = 1e-5):
                return x * self.ln_std + self.ln_mu

            self.run_time_activation_norm_fn_in = run_time_activation_ln_in
            self.run_time_activation_norm_fn_out = run_time_activation_ln_out
        else:
            self.run_time_activation_norm_fn_in = lambda x: x
            self.run_time_activation_norm_fn_out = lambda x: x



        self.activation_fn = get_activation_fn(
                self.cfg.activation_fn_str, **self.cfg.activation_fn_kwargs 
            ) 

        self.setup()  # Required for `HookedRootModule`s
        

    def initialize_weights_gated(self):
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
                )
            )
        )

        self.b_gate = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.r_mag = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.b_mag = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
        )

        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )


    def initialize_weights(self, out_features, in_features):
        """
        Initialize weights for the Sparse Autoencoder.
        
        This function uses Kaiming uniform initialization and then normalizes
        the weights to have unit normls
         along the output dimension.
        
        Args:
        out_features (int): Number of output features
        in_features (int): Number of input features
        
        Returns:
        torch.Tensor: Initialized weight matrix
        """
        weight = torch.empty(out_features, in_features, dtype=self.dtype, device=self.device)
        
        # Kaiming uniform initialization
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        
        # Normalize to unit norm along output dimension
        with torch.no_grad():
            weight /= torch.norm(weight, dim=1, keepdim=True)
        
        return weight
    

    def encode_gated(self, x: torch.Tensor):

        x = x.to(self.dtype)
        x = self.run_time_activation_norm_fn_in(x)

        sae_in = self.hook_sae_in(x - self.b_dec)

        # Gating path
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path with weight sharing
        magnitude_pre_activation = sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag

        feature_magnitudes = torch.relu(magnitude_pre_activation)

        feature_acts = self.hook_hidden_post(active_features * feature_magnitudes)

        return sae_in, feature_acts
    

    def encode_standard(self, x: torch.Tensor):
        # move x to correct dtype
        x = x.to(self.dtype)

        sae_in = self.run_time_activation_norm_fn_in(x)

        sae_in = self.hook_sae_in(
            sae_in - self.b_dec
        )  # Remove decoder bias as per Anthropic

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )
        feature_acts = self.hook_hidden_post(self.activation_fn(hidden_pre))

        return feature_acts


    from line_profiler import profile
    @profile
    def forward(self, x: torch.Tensor, dead_neuron_mask: torch.Tensor = None):
        if self.cfg.architecture == "standard":
            feature_acts = self.encode_standard(x)
        elif self.cfg.architecture == "gated":
            sae_in, feature_acts = self.encode_gated(x)
        else:
            raise ValueError(f"Architecture: {self.cfg.architecture} is not supported.")
        
        sae_out = self.hook_sae_out(
            einops.einsum(
                feature_acts,
                self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec
        )

        sae_out = self.run_time_activation_norm_fn_out(sae_out)

        # add config for whether l2 is normalized:
        x_centred = x - x.mean(dim=0, keepdim=True)

        mse_loss = torch.nn.functional.mse_loss(sae_out, x.detach(), reduction='none')
        norm_factor = torch.norm(x_centred, p=2, dim=-1, keepdim=True)
        mse_loss = mse_loss / norm_factor
 
        if self.zero_loss is None:
            self.zero_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        # gate on config and training so evals is not slowed down.
        if (
            self.cfg.use_ghost_grads
            and self.training
            and dead_neuron_mask is not None
            and  torch.any(dead_neuron_mask)
        ):
            # ghost protocol
            # 1.
            residual = x - sae_out
            residual_centred = residual - residual.mean(dim=0, keepdim=True)
            l2_norm_residual = torch.norm(residual, dim=-1)

            # 2.
            feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
            ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
            ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

            # 3.
            mse_loss_ghost_resid = (
                torch.pow((ghost_out - residual.detach().float()), 2)
                / (residual_centred.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
            )
            mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
            mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid

            mse_loss_ghost_resid = mse_loss_ghost_resid.mean()
        else:
            mse_loss_ghost_resid = self.zero_loss


        mse_loss = mse_loss.mean()
        sparsity = feature_acts.norm(p=self.lp_norm, dim=1).mean(dim=(0,))
        
        if self.cfg.architecture == "standard":
            aux_reconstruction_loss = torch.tensor(0.0)

            if self.cfg.activation_fn_str != "topk":
                l1_loss = self.l1_coefficient * sparsity
                loss = mse_loss + l1_loss + mse_loss_ghost_resid
            elif self.cfg.activation_fn_str == "topk": # Don't use L1 loss with topk
                l1_loss = None
                loss = mse_loss + mse_loss_ghost_resid

        elif self.cfg.architecture == "gated":
            pi_gate = sae_in @ self.W_enc + self.b_gate
            pi_gate_act = torch.relu(pi_gate)

            # SFN sparsity loss - summed over the feature dimension and averaged over the batch
            l1_loss = (
                self.l1_coefficient
                * torch.sum(pi_gate_act * self.W_dec.norm(dim=1), dim=-1).mean()
            )

            # Auxiliary reconstruction loss - summed over the feature dimension and averaged over the batch
            via_gate_reconstruction = pi_gate_act @ self.W_dec + self.b_dec
            aux_reconstruction_loss = torch.sum(
                (via_gate_reconstruction - sae_in) ** 2, dim=-1
            ).mean()

            loss = mse_loss + l1_loss + aux_reconstruction_loss


        return sae_out, feature_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid, aux_reconstruction_loss
    


    @torch.no_grad()
    def initialize_b_dec_with_precalculated(self, origin: torch.Tensor):
        out = torch.tensor(origin, dtype=self.dtype, device=self.device)
        self.b_dec.data = out

    @torch.no_grad()
    def initialize_b_dec(self, all_activations: torch.Tensor):
        if self.cfg.b_dec_init_method == "geometric_median":
            self.initialize_b_dec_with_geometric_median(all_activations)
        elif self.cfg.b_dec_init_method == "mean":
            self.initialize_b_dec_with_mean(all_activations)
        elif self.cfg.b_dec_init_method == "zeros":
            pass
        else:
            raise ValueError(
                f"Unexpected b_dec_init_method: {self.cfg.b_dec_init_method}"
            )

    @torch.no_grad()
    def initialize_b_dec_with_geometric_median(self, all_activations: torch.Tensor):
        previous_b_dec = self.b_dec.clone().cpu()
        out = compute_geometric_median(
            all_activations, skip_typechecks=True, maxiter=100, per_component=False
        ).median

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with geometric median of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        out = torch.tensor(out, dtype=self.dtype, device=self.device)
        self.b_dec.data = out

    @torch.no_grad()
    def initialize_b_dec_with_mean(self, all_activations: torch.Tensor):
        previous_b_dec = self.b_dec.clone().cpu()
        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with mean of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.b_dec.data = out.to(self.dtype).to(self.device)

    @torch.no_grad()
    def get_test_loss(self, batch_tokens: torch.Tensor, model: HookedRootModule):
        """
        A method for running the model with the SAE activations in order to return the loss.
        returns per token loss when activations are substituted in.
        """
        head_index = self.cfg.hook_point_head_index

        def standard_replacement_hook(activations: torch.Tensor, hook: Any):
            activations = self.forward(activations)[0].to(activations.dtype)
            return activations

        def head_replacement_hook(activations: torch.Tensor, hook: Any):
            new_actions = self.forward(activations[:, :, head_index])[0].to(
                activations.dtype
            )
            activations[:, :, head_index] = new_actions
            return activations

        replacement_hook = (
            standard_replacement_hook if head_index is None else head_replacement_hook
        )

        ce_loss_with_recons = model.run_with_hooks(
            batch_tokens,
            return_type="loss",
            fwd_hooks=[(self.cfg.hook_point, replacement_hook)],
        )

        return ce_loss_with_recons

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        assert parallel_component is not None  # keep pyright happy

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    def save_model(self, path: str):
        """
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        """

        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        state_dict = {"cfg": self.cfg, "state_dict": self.state_dict()}

        if path.endswith(".pt"):
            torch.save(state_dict, path)
        elif path.endswith("pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz"
            )

        print(f"Saved SAE to {path}")

    @classmethod
    def load_from_pretrained_legacy_saelens_v2(cls, path: str, current_cfg=None):
        """
        Load function for the model. Loads the model's state_dict and the config used to train it.
        This method can be called directly on the class, without needing an instance.
        """
        from vit_prisma.sae.sae_utils import map_legacy_sae_lens_2_to_prisma_repo
        

        # Ensure the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at specified path: {path}")

        # Load the state dictionary
        if path.endswith(".pt"):
            try:
                if torch.backends.mps.is_available():
                    state_dict = torch.load(path, map_location="mps")
                else:
                    state_dict = torch.load(path)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pt file: {e}")
        elif path.endswith(".pkl.gz"):
            try:
                with gzip.open(path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl.gz file: {e}")
        elif path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl file: {e}")
        else:
            raise ValueError(f"Unexpected file extension: {path}, supported extensions are .pt, .pkl, and .pkl.gz")

        # Extract the config and model state dict
        if 'config' in state_dict and 'autoencoder' in state_dict:
            old_config = state_dict['config']
            model_state_dict = state_dict['autoencoder']['state_dict']
        else:
            raise ValueError("The loaded state dictionary must contain 'config' and 'autoencoder' keys")
        
        
        mapped_config = map_legacy_sae_lens_2_to_prisma_repo(old_config)

        # Remove any fields that are not in VisionModelSAERunnerConfig
        valid_fields = set(field.name for field in fields(VisionModelSAERunnerConfig))
        mapped_config = {k: v for k, v in mapped_config.items() if k in valid_fields}


        config = VisionModelSAERunnerConfig(**mapped_config)


        # Update loaded config with current config if provided
        if current_cfg is not None:
            for key, value in vars(current_cfg).items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Fix the naming schema
        hook_point_layer = config.hook_point_layer
        if isinstance(hook_point_layer, list) and len(hook_point_layer) == 1:
            hook_point_layer = hook_point_layer[0]  # Extract the single integer

        # Construct the correct key
        hook_point = f'blocks.{hook_point_layer}.hook_mlp_out'
        config.hook_point = hook_point

        # Create an instance of the class using the loaded configuration
        instance = cls(cfg=config)
        instance.load_state_dict(model_state_dict)


        return instance

        # # Convert config to VisionModelSAERunnerConfig if necessary
        # if not isinstance(config, VisionModelSAERunnerConfig):
        #     config = VisionModelSAERunnerConfig(**config)

        # # Handle legacy issues
        # if not hasattr(config, "activation_fn_kwargs"):
        #     config.activation_fn_kwargs = {}

        # # Update loaded config with current config if provided
        # if current_cfg is not None:
        #     for key, value in vars(current_cfg).items():
        #         if hasattr(config, key):
        #             setattr(config, key, value)

        # # Create an instance of the class using the loaded configuration
        # instance = cls(cfg=config)
        # instance.load_state_dict(model_state_dict)

        # return instance

    @classmethod
    def load_from_pretrained(cls, weights_path, current_cfg=None):
        """
        Load function for the model. Can handle either:
        1. A single weights_path containing both config and weights (legacy format)
        2. Separate config_path and weights_path
        3. HuggingFace-style paths
        
        Args:
            weights_path (str): Path to weights file or HuggingFace repo ID
            config_path (str, optional): Path to config.json file. If None, will look for config in weights_path
            current_cfg: Optional configuration to override loaded settings
        """
        def load_config_from_json(config_path):
            """Helper to load and parse config from JSON."""
            from vit_prisma.sae.config import VisionModelSAERunnerConfig
            return VisionModelSAERunnerConfig.load_config(config_path)

        def load_weights(path, device=None):
            """Helper to load weights with appropriate device mapping."""
            if device:
                return torch.load(path, map_location=device)
            return torch.load(path)

        # Check if weights file exists
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"No weights file found at: {weights_path}")

        # Set device
        device = "mps" if torch.backends.mps.is_available() else None

        # Try loading weights file
        if weights_path.endswith(".pt"):
            try:
                state_dict = load_weights(weights_path, device)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pt file: {e}")
        elif weights_path.endswith(".pkl.gz"):
            try:
                with gzip.open(weights_path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading from .pkl.gz file: {e}")
        elif weights_path.endswith(".pkl"):
            try:
                with open(weights_path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading from .pkl file: {e}")
        else:
            raise ValueError(f"Unexpected file extension: {weights_path}")

        # Check if this is legacy format (combined config and weights)
        is_legacy = isinstance(state_dict, dict) and "cfg" in state_dict and "state_dict" in state_dict

        if is_legacy and config_path is None:
            # Use config from legacy file
            loaded_cfg = state_dict["cfg"]
            weights = state_dict["state_dict"]
        else:
            # Handle separate config and weights
                # Look for config.json in same directory as weights
            config_path = os.path.join(os.path.dirname(weights_path), "config.json")
            if not os.path.isfile(config_path):
                raise FileNotFoundError(
                    f"No config file found at {config_path} and no legacy format detected"
                )
        
            # Load config and weights separately
            loaded_cfg = load_config_from_json(config_path)
            weights = state_dict if not is_legacy else state_dict["state_dict"]

        # Set device in config if using MPS
        if device == "mps":
            loaded_cfg.device = "mps"

        # Handle legacy activation function kwargs
        if not hasattr(loaded_cfg, "activation_fn_kwargs"):
            if hasattr(loaded_cfg, "activation_fn_str"):
                if loaded_cfg.activation_fn_str == 'relu':
                    loaded_cfg.activation_fn_kwargs = {}
                elif loaded_cfg.activation_fn_str == 'leaky_relu':
                    loaded_cfg.activation_fn_kwargs = {'negative_slope': 0.01}
                else:
                    loaded_cfg.activation_fn_kwargs = {}
            else:
                loaded_cfg.activation_fn_kwargs = {}

        # Update loaded config with current config if provided
        if current_cfg is not None:
            for key, value in vars(current_cfg).items():
                if hasattr(loaded_cfg, key):
                    setattr(loaded_cfg, key, value)

        # Create and load model
        instance = cls(cfg=loaded_cfg)
        instance.load_state_dict(weights)

        return instance

    def get_name(self):
        sae_name = f"sparse_autoencoder_{self.cfg.model_name}_{self.cfg.hook_point}_{self.cfg.d_sae}"
        return sae_name

class TopK(nn.Module):
    def __init__(
        self, k: int, postact_fn: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()
    ):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result


def get_activation_fn(
    activation_fn: str, **kwargs: Any
) -> Callable[[torch.Tensor], torch.Tensor]:

    print(f"get_activation_fn received: activation_fn={activation_fn}, kwargs={kwargs}")

    if activation_fn == "relu":
        return torch.nn.ReLU()
    elif activation_fn == "tanh-relu":

        def tanh_relu(input: torch.Tensor) -> torch.Tensor:
            input = torch.relu(input)
            input = torch.tanh(input)
            return input
        return tanh_relu
    elif activation_fn == "topk":
        assert "k" in kwargs, "TopK activation function requires a k value."
        k = kwargs.get("k", 1)  # Default k to 1 if not provided
        postact_fn = kwargs.get(
            "postact_fn", nn.ReLU()
        )  # Default post-activation to ReLU if not provided
        return TopK(k, postact_fn)
    else:
        raise ValueError(f"Unknown activation function: {activation_fn}")