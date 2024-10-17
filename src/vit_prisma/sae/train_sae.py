import os
import re
import sys
import uuid
from dataclasses import is_dataclass, fields

import torch
import wandb
from torch.optim import Adam
from tqdm import tqdm

from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.evals.evaluator import Evaluator
from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.sae.sae_utils import wandb_log_suffix
from vit_prisma.sae.training.activations_store import VisionActivationsStore
from vit_prisma.sae.training.geometric_median import compute_geometric_median
from vit_prisma.sae.training.get_scheduler import get_scheduler
from vit_prisma.utils.data_utils.loader import load_dataset
from vit_prisma.utils.load_model import load_model


class VisionSAETrainer:
    def __init__(self, cfg: VisionModelSAERunnerConfig):
        self.cfg = cfg

        self.set_default_attributes() # For backward compatability

        self.bad_run_check = True if self.cfg.min_l0 and self.cfg.min_explained_variance else False
        self.model = load_model(self.cfg)
        self.sae = SparseAutoencoder(self.cfg)

        dataset, eval_dataset, visualize_eval_dataset = load_dataset(self.cfg, visualize=True)
        self.activations_store = self.initialize_activations_store(dataset, eval_dataset)

        self.cfg.wandb_project = self.cfg.model_name.replace('/', '-') + "-expansion-" + str(self.cfg.expansion_factor) + "-layer-" + str(self.cfg.hook_point_layer)
        self.cfg.unique_hash = uuid.uuid4().hex[:8]  # Generate a random 8-character hex string
        self.cfg.run_name = self.cfg.unique_hash + "-" + self.cfg.wandb_project

        self.checkpoint_thresholds = self.get_checkpoint_thresholds()
        self.setup_checkpoint_path()

        self.cfg.pretty_print() if self.cfg.verbose else None

        self.evaluator = Evaluator(self.model, eval_dataset, self.cfg, visualize_eval_dataset)

    def set_default_attributes(self):
        '''
        For backward compatability, add new attributes here
        '''
        # Set default values for attributes that might not be in the loaded config
        default_attributes = [
            'min_l0', 'min_explained_variance']
    
        for attr in default_attributes:
            if not hasattr(self.cfg, attr):
                setattr(self.cfg, attr, None)
        

    def setup_checkpoint_path(self):
        # Create checkpoint path with run_name, which contains unique identifier
        self.cfg.checkpoint_path = f"{self.cfg.checkpoint_path}/{self.cfg.run_name}"
        os.makedirs(self.cfg.checkpoint_path, exist_ok=True)
        print(f"Checkpoint path: {self.cfg.checkpoint_path}") if self.cfg.verbose else None

    def initialize_activations_store(self, dataset, eval_dataset):
        # raise separate errors if dataset or eval_dataset is none or invalid format. instead of none, do dataset type
        if dataset is None:
            raise ValueError("Training dataset is None")
        if eval_dataset is None:
            raise ValueError("Eval dataset is None")
        return VisionActivationsStore(self.cfg, self.model, dataset, eval_dataset=eval_dataset)
    
    def get_checkpoint_thresholds(self):
        """These are the steps at which to save the checkpoints of the SAE."""

        if self.cfg.n_checkpoints > 0:
            return list(range(0, self.cfg.total_training_tokens, self.cfg.total_training_tokens // self.cfg.n_checkpoints))[1:]
        return []

    def initialize_training_variables(self):
        # num_saes = len(self.sae_group)
        act_freq_scores = torch.zeros(int(self.cfg.d_sae), device = self.cfg.device)
        n_forward_passes_since_fired = torch.zeros(int(self.cfg.d_sae), device=self.cfg.device)
        n_frac_active_tokens = 0 
        optimizers = Adam(self.sae.parameters(), lr=self.cfg.lr)
        scheduler = get_scheduler(
                        self.cfg.lr_scheduler_name,
                        optimizer=optimizers,
                        warm_up_steps=self.cfg.lr_warm_up_steps,
                        training_steps=self.cfg.total_training_steps,
                        lr_end=self.cfg.lr / 10,
                )       
        return act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens, optimizers, scheduler

    def initialize_geometric_medians(self):
        all_layers = self.sae.cfg.hook_point_layer
        geometric_medians = {}
        if not isinstance(all_layers, list):
            all_layers = [all_layers]
        hyperparams = self.sae.cfg
        sae_layer_id = all_layers.index(hyperparams.hook_point_layer)
        if hyperparams.b_dec_init_method == "geometric_median":
            layer_acts = self.activations_store.storage_buffer.detach()[:, sae_layer_id, :]
            if sae_layer_id not in geometric_medians:
                median = compute_geometric_median(layer_acts, maxiter=200).median
                geometric_medians[sae_layer_id] = median
            self.sae.initialize_b_dec_with_precalculated(geometric_medians[sae_layer_id])
        elif hyperparams.b_dec_init_method == "mean":
            layer_acts = self.activations_store.storage_buffer.detach().cpu()[:, sae_layer_id, :]
            self.sae.initialize_b_dec_with_mean(layer_acts)
        self.sae.train()
        return geometric_medians
    
    def train_step(self, sparse_autoencoder, optimizer, scheduler, act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens, 
               layer_acts, n_training_steps, n_training_tokens):
    
        hyperparams = sparse_autoencoder.cfg

        all_layers = hyperparams.hook_point_layer if isinstance(hyperparams.hook_point_layer, list) else [hyperparams.hook_point_layer]
        layer_id = all_layers.index(hyperparams.hook_point_layer)
        sae_in = layer_acts[:, layer_id, :]

        sparse_autoencoder.train()
        sparse_autoencoder.set_decoder_norm_to_unit_norm()

        # Log feature sparsity every feature_sampling_window steps
        if (n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()

            if self.cfg.log_to_wandb:
                self._log_feature_sparsity(sparse_autoencoder, hyperparams, log_feature_sparsity, feature_sparsity, n_training_steps)

            act_freq_scores = torch.zeros(sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device)
            n_frac_active_tokens = 0

        scheduler.step()
        optimizer.zero_grad()

        ghost_grad_neuron_mask = (n_forward_passes_since_fired > sparse_autoencoder.cfg.dead_feature_window).bool()

        # Forward and Backward Passes
        sae_out, feature_acts, loss, mse_loss, l1_loss, ghost_grad_loss = sparse_autoencoder(sae_in, ghost_grad_neuron_mask)
        
        with torch.no_grad():
            did_fire = (feature_acts > 0).float().sum(-2) > 0
            n_forward_passes_since_fired += 1
            n_forward_passes_since_fired[did_fire] = 0

            act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            batch_size = sae_in.shape[0]
            n_frac_active_tokens += batch_size 

            l0 = (feature_acts > 0).float().sum(-1).mean()

            if self.cfg.log_to_wandb and ((n_training_steps + 1) % self.cfg.wandb_log_frequency == 0):
                self._log_metrics(sparse_autoencoder, hyperparams, optimizer, sae_in, sae_out, n_forward_passes_since_fired, 
                                ghost_grad_neuron_mask, mse_loss, l1_loss, ghost_grad_loss, loss, l0, n_training_steps, n_training_tokens)

            if self.cfg.log_to_wandb and self.cfg.training_eval.log_frequency and ((n_training_steps + 1) % self.cfg.training_eval.log_frequency == 0):
                self.evaluator.evaluate(sparse_autoencoder, context="training")

        loss.backward()
        
        if self.cfg.max_grad_norm: # Gradient clipping 
            torch.nn.utils.clip_grad_norm_(sparse_autoencoder.parameters(), max_norm=self.cfg.max_grad_norm)

        sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        
        return loss, mse_loss, l1_loss, l0, act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens

    # def _log_feature_sparsity(self, sparse_autoencoder, hyperparams, log_feature_sparsity, feature_sparsity, n_training_steps):
    #     suffix = wandb_log_suffix(sparse_autoencoder.cfg, hyperparams)
    #     wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
    #     wandb.log({
    #         f"metrics/mean_log10_feature_sparsity{suffix}": log_feature_sparsity.mean().item(),
    #         f"plots/feature_density_line_chart{suffix}": wandb_histogram,
    #         f"sparsity/below_1e-5{suffix}": (feature_sparsity < 1e-5).sum().item(),
    #         f"sparsity/below_1e-6{suffix}": (feature_sparsity < 1e-6).sum().item(),
    #     }, step=n_training_steps)

    def _log_feature_sparsity(self, sparse_autoencoder, hyperparams, log_feature_sparsity, feature_sparsity, n_training_steps):
        suffix = wandb_log_suffix(sparse_autoencoder.cfg, hyperparams)
        
        # Original feature-level sparsity calculations
        log_sparsity_np = log_feature_sparsity.detach().cpu().numpy()
        log_sparsity_histogram = wandb.Histogram(log_sparsity_np)
        
        # # Calculate 1/sparsity values for feature-level
        # inverse_sparsity_np = 10 ** (-log_sparsity_np)
        # hist, bin_edges = np.histogram(np.log10(inverse_sparsity_np), bins=50)
        # total_tokens = len(inverse_sparsity_np)
        # proportion = hist / total_tokens
        # x_labels = 10 ** ((bin_edges[:-1] + bin_edges[1:]) / 2)
        
        # # Create custom wandb chart for inverse feature-level sparsity
        # data = [[x, y] for (x, y) in zip(x_labels, proportion)]
        # table = wandb.Table(data=data, columns=["1/Sparsity", "Proportion of Tokens"])
        # inverse_sparsity_chart = wandb.plot.bar(table, 
        #                                         "1/Sparsity", 
        #                                         "Proportion of Tokens",
        #                                         title="Inverse Feature Density Distribution")

         # New image-level sparsity calculations
        # total_tokens = feature_sparsity.shape[0]
        # n_features = feature_sparsity.shape[1] if len(feature_sparsity.shape) > 1 else 1
        
        # # Adjust total_tokens to discard remainder, but don't exceed original value
        # remainder = total_tokens % self.cfg.context_size
        # total_tokens_adjusted = total_tokens - remainder

        # n_images = total_tokens_adjusted // self.cfg.context_size
        # reshaped_sparsity = feature_sparsity[:total_tokens_adjusted].view(n_images, self.cfg.context_size, n_features)
        # per_image_sparsity = reshaped_sparsity.mean(dim=(1, 2))
        # per_image_log_sparsity = torch.log10(per_image_sparsity)
        # per_image_log_sparsity_np = per_image_log_sparsity.detach().cpu().numpy()

        # # Create wandb Histogram for image-level log sparsity (matching original format)
        # image_log_sparsity_histogram = wandb.Histogram(per_image_log_sparsity_np)

        wandb.log({
            # Original metrics
            f"metrics/mean_log10_feature_sparsity{suffix}": log_feature_sparsity.mean().item(),
            f"plots/log_feature_density_histogram{suffix}": log_sparsity_histogram,
            # f"plots/inverse_feature_density_histogram{suffix}": inverse_sparsity_chart,
            f"sparsity/below_1e-5{suffix}": (feature_sparsity < 1e-5).sum().item(),
            f"sparsity/below_1e-6{suffix}": (feature_sparsity < 1e-6).sum().item(),
            
            # # New image-level metrics
            # f"metrics/mean_log10_per_image_sparsity{suffix}": per_image_log_sparsity.mean().item(),
            # f"plots/log_per_image_sparsity_histogram{suffix}": image_log_sparsity_histogram,
            # f"sparsity/images_below_1e-5{suffix}": (per_image_sparsity < 1e-5).sum().item(),
            # f"sparsity/images_below_1e-6{suffix}": (per_image_sparsity < 1e-6).sum().item(),
        }, step=n_training_steps)

    def _log_metrics(self, sparse_autoencoder, hyperparams, optimizer, sae_in, sae_out, n_forward_passes_since_fired, 
                    ghost_grad_neuron_mask, mse_loss, l1_loss, ghost_grad_loss, loss, l0, n_training_steps, n_training_tokens):
        current_learning_rate = optimizer.param_groups[0]["lr"]
        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance = 1 - per_token_l2_loss / total_variance

        if (self.bad_run_check) and (l0.item()) < self.cfg.min_l0 and (explained_variance.mean().item() < self.cfg.min_explained_variance):
            print(f"Skipping bad run. Moving to the next run.")
            wandb.finish()
            sys.exit()

        n_training_images = n_training_tokens // self.cfg.context_size

        if l1_loss is None: # When using top k SAE loss
            l1_loss = torch.tensor(0.0)

        suffix = wandb_log_suffix(sparse_autoencoder.cfg, hyperparams)
        metrics = {
            f"losses/mse_loss{suffix}": mse_loss.item(),
            f"losses/l1_loss{suffix}": l1_loss.item() / sparse_autoencoder.l1_coefficient,
            f"losses/ghost_grad_loss{suffix}": ghost_grad_loss.item(),
            f"losses/overall_loss{suffix}": loss.item(),
            f"metrics/explained_variance{suffix}": explained_variance.mean().item(),
            f"metrics/explained_variance_std{suffix}": explained_variance.std().item(),
            f"metrics/l0{suffix}": l0.item(),
            f"sparsity/mean_passes_since_fired{suffix}": n_forward_passes_since_fired.mean().item(),
            f"sparsity/dead_features{suffix}": ghost_grad_neuron_mask.sum().item(),
            f"details/current_learning_rate{suffix}": current_learning_rate,
            "details/n_training_tokens": n_training_tokens,
            "details/n_training_images": n_training_images,
        }
        wandb.log(metrics, step=n_training_steps)

    def log_metrics(self, sae, hyperparams, metrics, n_training_steps):
        if self.cfg.log_to_wandb and ((n_training_steps + 1) % self.cfg.wandb_log_frequency == 0):
            suffix = wandb_log_suffix(self.cfg, hyperparams)
            wandb.log(metrics, step=n_training_steps)

    def checkpoint(self, sae, n_training_tokens, act_freq_scores, n_frac_active_tokens):
        # NOTE fix htis code to not be sae groups anymore
        # path = f"{sae_group.cfg.checkpoint_path}/{n_training_images}_{sae_group.get_name()}.pt"

        n_training_images = n_training_tokens // self.cfg.context_size
        path = self.cfg.checkpoint_path + f"/n_images_{n_training_images}.pt"
        sae.set_decoder_norm_to_unit_norm()
        sae.save_model(path)

        # Save log feature sparsity
        log_feature_sparsity_path = self.cfg.checkpoint_path + f"/n_images_{n_training_images}_log_feature_sparsity.pt"
        feature_sparsity = (
            act_freq_scores / n_frac_active_tokens
        )
        log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()
        torch.save(log_feature_sparsity, log_feature_sparsity_path)

        if len(self.checkpoint_thresholds) != 0:
            self.checkpoint_thresholds.pop(0)

        if self.cfg.log_to_wandb:
            hyperparams = sae.cfg
            self.save_to_wandb(sae, hyperparams, path, log_feature_sparsity_path)

    def save_to_wandb(self, sae, hyperparams, path, log_feature_sparsity_path):
        suffix = wandb_log_suffix(sae.cfg, hyperparams)
        name_for_log = re.sub(self.cfg.unique_hash, '_', suffix)
        try:
            model_artifact = wandb.Artifact(
                f"{name_for_log}",
                type="model",
                metadata=dict(sae.cfg.__dict__),
            )
            model_artifact.add_file(path)
            wandb.log_artifact(model_artifact)

            sparsity_artifact = wandb.Artifact(
                f"{name_for_log}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(sae.cfg.__dict__),
            )
            sparsity_artifact.add_file(log_feature_sparsity_path)
            wandb.log_artifact(sparsity_artifact)
        except:
            pass

    @staticmethod
    def dataclass_to_dict(obj):
        if not is_dataclass(obj):
            return obj
        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            if is_dataclass(value):
                result[field.name] = VisionSAETrainer.dataclass_to_dict(value)
            else:
                result[field.name] = value
        return result
        
    def run(self):
        if self.cfg.log_to_wandb:
            config_dict = self.dataclass_to_dict(self.cfg)
            wandb.init(project=self.cfg.wandb_project, config=config_dict, name=self.cfg.run_name)

        act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens, optimizer, scheduler = self.initialize_training_variables()
        geometric_medians = self.initialize_geometric_medians()

        print("Starting training") if self.cfg.verbose else None

        n_training_steps = 0
        n_training_tokens = 0
        
        pbar = tqdm(total=self.cfg.total_training_tokens, desc="Training SAE")        
        while n_training_tokens < self.cfg.total_training_tokens:
            layer_acts = self.activations_store.next_batch()

            # init these here to avoid uninitialized vars
            mse_loss = torch.tensor(0.0)
            l1_loss = torch.tensor(0.0)

            loss, mse_loss, l1_loss, l0, act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens = self.train_step(
            sparse_autoencoder=self.sae,
            optimizer=optimizer,
            scheduler=scheduler,
            layer_acts=layer_acts,
            n_training_steps=n_training_steps,
            n_training_tokens=n_training_tokens,
            act_freq_scores=act_freq_scores,
            n_forward_passes_since_fired=n_forward_passes_since_fired,
            n_frac_active_tokens=n_frac_active_tokens)

            n_training_steps += 1
            n_training_tokens += self.cfg.train_batch_size

            if len(self.checkpoint_thresholds) > 0 and n_training_tokens > self.checkpoint_thresholds[0]:
                self.checkpoint(self.sae, n_training_tokens, act_freq_scores, n_frac_active_tokens)
                print(f"Checkpoint saved at {n_training_tokens} tokens") if self.cfg.verbose else None

            pbar.update(self.cfg.train_batch_size)

            if l1_loss is None: # When using top k SAE loss
                l1_loss = torch.tensor(0.0)
        
            pbar.set_description(
                f"Training SAE: Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, L1 Loss: {l1_loss.item():.4f}, L0: {l0:.4f}")
        
        # Final checkpoint
        self.checkpoint(self.sae, n_training_tokens, act_freq_scores, n_frac_active_tokens)
        print(f"Final checkpoint saved at {n_training_tokens} tokens") if self.cfg.verbose else None

        if self.cfg.post_training_eval.log_frequency:
            self.evaluator.evaluate(self.sae, context="post-training")

        pbar.close()

        return self.sae

    