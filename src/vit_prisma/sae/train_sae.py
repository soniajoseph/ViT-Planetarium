from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_10
from vit_prisma.utils.load_model import load_model
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.sae import StandardSparseAutoencoder, GatedSparseAutoencoder
from vit_prisma.sae.training.activations_store import VisionActivationsStore, CacheVisionActivationStore

import wandb

from vit_prisma.sae.training.geometric_median import compute_geometric_median
from vit_prisma.sae.training.get_scheduler import get_scheduler

# from vit_prisma.sae.evals import run_evals_vision
# from vit_prisma.sae.evals.evals import get_substitution_loss, get_text_embeddings, get_text_embeddings_openclip, get_text_labels

from vit_prisma.dataloaders.imagenet_index import imagenet_index

import torch
from torch.optim import Adam
from tqdm import tqdm
import re

# this should be abstracted out of this file in the long term
import open_clip

import os
import sys

import torchvision

import einops
import numpy as np

from typing import Any, cast

from dataclasses import is_dataclass, fields

import uuid

import wandb


def wandb_log_suffix(cfg: Any, hyperparams: Any):
    # Create a mapping from cfg list keys to their corresponding hyperparams attributes
    key_mapping = {
        "hook_point_layer": "layer",
        "l1_coefficient": "coeff",
        "lp_norm": "l",
        "lr": "lr",
    }

    # Generate the suffix by iterating over the keys that have list values in cfg
    suffix = "".join(
        f"_{key_mapping.get(key, key)}{getattr(hyperparams, key, '')}"
        for key, value in vars(cfg).items()
        if isinstance(value, list)
    )
    return suffix


class VisionSAETrainer:
    def __init__(self, cfg: VisionModelSAERunnerConfig):
        self.cfg = cfg

        self.set_default_attributes()  # For backward compatability

        self.bad_run_check = (
            True if self.cfg.min_l0 and self.cfg.min_explained_variance else False
        )
        self.model = load_model(self.cfg, self.cfg.model_name)

        if self.cfg.architecture == "gated":
            self.sae = GatedSparseAutoencoder(self.cfg)
        elif self.cfg.architecture == "standard" or self.cfg.architecture == "vanilla":
            self.sae = StandardSparseAutoencoder(self.cfg)
        else:
            raise ValueError(f"Loading of {self.cfg.architecture} not supported")


        dataset, eval_dataset = self.load_dataset()
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.activations_store = self.initialize_activations_store(
            dataset, eval_dataset
        )
        if not self.cfg.wandb_project:
            self.cfg.wandb_project = (
                self.cfg.model_name.replace("/", "-")
                + "-expansion-"
                + str(self.cfg.expansion_factor)
                + "-layer-"
                + str(self.cfg.hook_point_layer)
            )
        self.cfg.unique_hash = uuid.uuid4().hex[
            :8
        ]  # Generate a random 8-character hex string
        self.cfg.run_name = self.cfg.unique_hash + "-" + self.cfg.wandb_project

        self.checkpoint_thresholds = self.get_checkpoint_thresholds()
        self.setup_checkpoint_path()

        self.cfg.pretty_print() if self.cfg.verbose else None

    def set_default_attributes(self):
        """
        For backward compatability, add new attributes here
        """
        # Set default values for attributes that might not be in the loaded config
        default_attributes = ["min_l0", "min_explained_variance"]

        for attr in default_attributes:
            if not hasattr(self.cfg, attr):
                setattr(self.cfg, attr, None)

    def setup_checkpoint_path(self):
        if self.cfg.n_checkpoints:
            # Create checkpoint path with run_name, which contains unique identifier
            self.cfg.checkpoint_path = f"{self.cfg.checkpoint_path}/{self.cfg.run_name}"
            os.makedirs(self.cfg.checkpoint_path, exist_ok=True)
            (
                print(f"Checkpoint path: {self.cfg.checkpoint_path}")
                if self.cfg.verbose
                else None
            )
        else:
            print(f"Not saving checkpoints so skipping creating checkpoint directory")

    def initialize_activations_store(self, dataset, eval_dataset):
        # raise separate errors if dataset or eval_dataset is none or invalid format. instead of none, do dataset type
        if dataset is None:
            raise ValueError("Training dataset is None")
        if eval_dataset is None:
            raise ValueError("Eval dataset is None")
        

        if self.cfg.use_cached_activations:
            return CacheVisionActivationStore(self.cfg)

        return VisionActivationsStore(
            self.cfg,
            self.model,
            dataset,
            eval_dataset=eval_dataset,
            num_workers=self.cfg.num_workers,
        )

    def load_dataset(self, model_type="clip"):
        if self.cfg.dataset_name == "imagenet1k":
            (
                print(f"Dataset type: {self.cfg.dataset_name}")
                if self.cfg.verbose
                else None
            )
            # Imagenet-specific logic
            from vit_prisma.utils.data_utils.imagenet.imagenet_utils import (
                setup_imagenet_paths,
            )
            from vit_prisma.dataloaders.imagenet_dataset import (
                ImageNetValidationDataset,
            )

            if model_type == "clip":
                from vit_prisma.transforms.open_clip_transforms import (
                    get_clip_val_transforms,
                )

                data_transforms = get_clip_val_transforms(self.cfg.image_size)
            else:
                raise ValueError("Invalid model type")
            imagenet_paths = setup_imagenet_paths(self.cfg.dataset_path)

            train_data = torchvision.datasets.ImageFolder(
                self.cfg.dataset_train_path, transform=data_transforms
            )
            val_data = ImageNetValidationDataset(
                self.cfg.dataset_val_path,
                imagenet_paths["label_strings"],
                imagenet_paths["val_labels"],
                data_transforms,
            )
            print(f"Train data length: {len(train_data)}") if self.cfg.verbose else None
            (
                print(f"Validation data length: {len(val_data)}")
                if self.cfg.verbose
                else None
            )
            return train_data, val_data
        elif self.cfg.dataset_name == "cifar10":
            train_data, val_data, test_data = load_cifar_10(
                self.cfg.dataset_path, image_size=self.cfg.image_size
            )
            print(f"Train data length: {len(train_data)}") if self.cfg.verbose else None
            print(f"Validation data length: {len(val_data)}") if self.cfg.verbose else None
            return train_data, val_data
        else:
            # raise error
            raise ValueError("Invalid dataset name")

    def get_checkpoint_thresholds(self):
        if self.cfg.n_checkpoints > 0:
            return list(
                range(
                    0,
                    self.cfg.total_training_tokens,
                    self.cfg.total_training_tokens // self.cfg.n_checkpoints,
                )
            )[1:]
        return []

    def initialize_training_variables(self):
        # num_saes = len(self.sae_group)
        act_freq_scores = torch.zeros(int(self.cfg.d_sae), device=self.cfg.device)
        n_forward_passes_since_fired = torch.zeros(
            int(self.cfg.d_sae), device=self.cfg.device
        )
        n_frac_active_tokens = 0
        optimizers = Adam(self.sae.parameters(), lr=self.cfg.lr)
        scheduler = get_scheduler(
            self.cfg.lr_scheduler_name,
            optimizer=optimizers,
            warm_up_steps=self.cfg.lr_warm_up_steps,
            training_steps=self.cfg.total_training_steps,
            lr_end=self.cfg.lr / 10,
        )
        return (
            act_freq_scores,
            n_forward_passes_since_fired,
            n_frac_active_tokens,
            optimizers,
            scheduler,
        )

    def initialize_geometric_medians(self):
        all_layers = self.sae.cfg.hook_point_layer
        geometric_medians = {}
        if not isinstance(all_layers, list):
            all_layers = [all_layers]
        hyperparams = self.sae.cfg
        sae_layer_id = all_layers.index(hyperparams.hook_point_layer)
        if hyperparams.b_dec_init_method == "geometric_median":
            layer_acts = self.activations_store.storage_buffer.detach()[
                :, sae_layer_id, :
            ]
            if sae_layer_id not in geometric_medians:
                median = compute_geometric_median(layer_acts, maxiter=200).median
                geometric_medians[sae_layer_id] = median
            self.sae.initialize_b_dec_with_precalculated(
                geometric_medians[sae_layer_id]
            )
        elif hyperparams.b_dec_init_method == "mean":
            layer_acts = self.activations_store.storage_buffer.detach().cpu()[
                :, sae_layer_id, :
            ]
            self.sae.initialize_b_dec_with_mean(layer_acts)
        self.sae.train()
        return geometric_medians

    def train_step(
        self,
        sparse_autoencoder,
        optimizer,
        scheduler,
        act_freq_scores,
        n_forward_passes_since_fired,
        n_frac_active_tokens,
        layer_acts,
        n_training_steps,
        n_training_tokens,
    ):

        hyperparams = sparse_autoencoder.cfg

        all_layers = (
            hyperparams.hook_point_layer
            if isinstance(hyperparams.hook_point_layer, list)
            else [hyperparams.hook_point_layer]
        )
        layer_id = all_layers.index(hyperparams.hook_point_layer)
        sae_in = layer_acts[:, layer_id, :]

        sparse_autoencoder.train()
        sparse_autoencoder.set_decoder_norm_to_unit_norm()

        # Log feature sparsity every feature_sampling_window steps
        if (n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()

            if self.cfg.log_to_wandb:
                self._log_feature_sparsity(
                    sparse_autoencoder,
                    hyperparams,
                    log_feature_sparsity,
                    feature_sparsity,
                    n_training_steps,
                )

            act_freq_scores = torch.zeros(
                sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device
            )
            n_frac_active_tokens = 0

        optimizer.zero_grad()

        ghost_grad_neuron_mask = (
            n_forward_passes_since_fired > sparse_autoencoder.cfg.dead_feature_window
        ).bool()

        # Forward and Backward Passes
        (
            sae_out,
            feature_acts,
            loss,
            mse_loss,
            l1_loss,
            ghost_grad_loss,
            aux_reconstruction_loss,
        ) = sparse_autoencoder(sae_in, ghost_grad_neuron_mask)

        with torch.no_grad():
            did_fire = (feature_acts > 0).float().sum(-2) > 0
            n_forward_passes_since_fired += 1
            n_forward_passes_since_fired[did_fire] = 0

            act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            batch_size = sae_in.shape[0]
            n_frac_active_tokens += batch_size

            l0 = (feature_acts > 0).float().sum(-1).mean()

            if self.cfg.log_to_wandb and (
                (n_training_steps + 1) % self.cfg.wandb_log_frequency == 0
            ):
                self._log_metrics(
                    sparse_autoencoder,
                    hyperparams,
                    optimizer,
                    sae_in,
                    sae_out,
                    n_forward_passes_since_fired,
                    ghost_grad_neuron_mask,
                    mse_loss,
                    l1_loss,
                    aux_reconstruction_loss,
                    ghost_grad_loss,
                    loss,
                    l0,
                    n_training_steps,
                    n_training_tokens,
                )

            # if self.cfg.log_to_wandb and ((n_training_steps + 1) % (self.cfg.wandb_log_frequency * 10) == 0):
            #     self._run_evals(sparse_autoencoder, hyperparams, n_training_steps)

        loss.backward()

        if self.cfg.max_grad_norm:  # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                sparse_autoencoder.parameters(), max_norm=self.cfg.max_grad_norm
            )

        sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        scheduler.step()

        return (
            loss,
            mse_loss,
            l1_loss,
            l0,
            act_freq_scores,
            n_forward_passes_since_fired,
            n_frac_active_tokens,
        )

    # layer_acts be a poor format - need to run in ctx_len, gt_labels format
    @torch.no_grad()
    def val(self, sparse_autoencoder):
        sparse_autoencoder.eval()

        print("Running validation") if self.cfg.verbose else None

        for images, gt_labels in self.activations_store.image_dataloader_eval:
            images = images.to(self.cfg.device)
            gt_labels = gt_labels.to(self.cfg.device)
            # needs to start with batch_size dimension
            _, cache = self.model.run_with_cache(
                images, names_filter=sparse_autoencoder.cfg.hook_point
            )
            hook_point_activation = cache[sparse_autoencoder.cfg.hook_point].to(
                self.cfg.device
            )

            sae_out, feature_acts, loss, mse_loss, l1_loss, _, _ = sparse_autoencoder(
                hook_point_activation
            )

            # explained variance
            sae_in = hook_point_activation
            per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
            total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
            explained_variance = 1 - per_token_l2_loss / total_variance

            # L0
            l0 = (feature_acts > 0).float().sum(-1).mean()

            # Calculate cosine similarity between original activations and sae output
            cos_sim = (
                torch.cosine_similarity(
                    einops.rearrange(
                        hook_point_activation, "batch seq d_mlp -> (batch seq) d_mlp"
                    ),
                    einops.rearrange(sae_out, "batch seq d_mlp -> (batch seq) d_mlp"),
                    dim=0,
                )
                .mean(-1)
                .tolist()
            )
            # all_cosine_similarity.append(cos_sim)

            # Calculate substitution loss
            # we need to get the imagenet labels of all the images in our batch
            # just map gt_label to imagenet name

            # this should only run if this is a clip model
            if self.cfg.model_name.startswith("open-clip:"):
                # create a list of all imagenet classes
                num_imagenet_classes = 1000
                batch_label_names = [
                    imagenet_index[str(int(label))][1]
                    for label in range(num_imagenet_classes)
                ]

                model_name = (
                    self.cfg.model_name
                    if not self.cfg.model_name.startswith("open-clip:")
                    else self.cfg.model_name[10:]
                )
                print(f"model_name: {model_name}")
                # bad bad bad
                oc_model_name = "hf-hub:" + model_name

                # should be moved to hookedvit pretrained long terms
                og_model, _, preproc = open_clip.create_model_and_transforms(
                    oc_model_name
                )
                tokenizer = open_clip.get_tokenizer("ViT-B-32")

                text_embeddings = get_text_embeddings_openclip(
                    og_model, preproc, tokenizer, batch_label_names
                )
                # print(f"text_embeddings: {text_embeddings.shape}")
                score, model_loss, sae_recon_loss, zero_abl_loss = (
                    get_substitution_loss(
                        sparse_autoencoder,
                        self.model,
                        images,
                        gt_labels,
                        text_embeddings,
                        device=self.cfg.device,
                    )
                )
                # log to w&b
                # print(f"score: {score}")
                # print(f"loss: {model_loss}")
                # print(f"subst_loss: {sae_recon_loss}")
                # print(f"zero_abl_loss: {zero_abl_loss}")

            # count += 1
            # if count > self.cfg.max_val_points:
            #     break

            # print(f"sae loss: {loss}")
            # print(f"sae loss.shape: {loss.shape}")
            # print(f"mse_loss: {mse_loss}")
            # print(f"l1_loss: {l1_loss}")
            # print(f"l1_loss.shape: {l1_loss.shape}")

            wandb.log(
                {
                    # Original metrics
                    f"validation_metrics/mse_loss": mse_loss,
                    f"validation_metrics/substitution_score": score,
                    f"validation_metrics/substitution_loss": sae_recon_loss,
                    f"validation_metrics/explained_variance": explained_variance.mean().item(),
                    f"validation_metrics/L0": l0,
                    # # New image-level metrics
                    # f"metrics/mean_log10_per_image_sparsity{suffix}": per_image_log_sparsity.mean().item(),
                    # f"plots/log_per_image_sparsity_histogram{suffix}": image_log_sparsity_histogram,
                    # f"sparsity/images_below_1e-5{suffix}": (per_image_sparsity < 1e-5).sum().item(),
                    # f"sparsity/images_below_1e-6{suffix}": (per_image_sparsity < 1e-6).sum().item(),
                }
            )


            # log to w&b
            print(f"cos_sim: {cos_sim}")
            break  # Currently runs just one batch for efficiency

    # def _log_feature_sparsity(self, sparse_autoencoder, hyperparams, log_feature_sparsity, feature_sparsity, n_training_steps):
    #     suffix = wandb_log_suffix(sparse_autoencoder.cfg, hyperparams)
    #     wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
    #     wandb.log({
    #         f"metrics/mean_log10_feature_sparsity{suffix}": log_feature_sparsity.mean().item(),
    #         f"plots/feature_density_line_chart{suffix}": wandb_histogram,
    #         f"sparsity/below_1e-5{suffix}": (feature_sparsity < 1e-5).sum().item(),
    #         f"sparsity/below_1e-6{suffix}": (feature_sparsity < 1e-6).sum().item(),
    #     }, step=n_training_steps)

    def _log_feature_sparsity(
        self,
        sparse_autoencoder,
        hyperparams,
        log_feature_sparsity,
        feature_sparsity,
        n_training_steps,
    ):
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

        wandb.log(
            {
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
            },
            step=n_training_steps,
        )

    def _log_metrics(
        self,
        sparse_autoencoder,
        hyperparams,
        optimizer,
        sae_in,
        sae_out,
        n_forward_passes_since_fired,
        ghost_grad_neuron_mask,
        mse_loss,
        l1_loss,
        aux_reconstruction_loss,
        ghost_grad_loss,
        loss,
        l0,
        n_training_steps,
        n_training_tokens,
    ):
        current_learning_rate = optimizer.param_groups[0]["lr"]
        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance = 1 - per_token_l2_loss / total_variance

        if (
            (self.bad_run_check)
            and (l0.item()) < self.cfg.min_l0
            and (explained_variance.mean().item() < self.cfg.min_explained_variance)
        ):
            print(f"Skipping bad run. Moving to the next run.")
            wandb.finish()
            sys.exit()

        n_training_images = n_training_tokens // self.cfg.context_size

        if l1_loss is None:  # When using top k SAE loss
            l1_loss = torch.tensor(0.0)

        suffix = wandb_log_suffix(sparse_autoencoder.cfg, hyperparams)
        metrics = {
            f"losses/mse_loss{suffix}": mse_loss.item(),
            f"losses/l1_loss{suffix}": l1_loss.item()
            / sparse_autoencoder.l1_coefficient,
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

        if self.cfg.architecture == "gated":
            metrics[f"losses/aux_reconstruction_loss{suffix}"] = (
                aux_reconstruction_loss.item()
            )

        wandb.log(metrics, step=n_training_steps)

    def _run_evals(self, sparse_autoencoder, hyperparams, n_training_steps):
        sparse_autoencoder.eval()
        suffix = wandb_log_suffix(sparse_autoencoder.cfg, hyperparams)
        try:
            run_evals_vision(
                sparse_autoencoder,
                self.activations_store,
                self.model,
                n_training_steps,
                suffix=suffix,
            )
        except Exception as e:
            print(f"Error in run_evals_vision: {e}")
        sparse_autoencoder.train()

    def log_metrics(self, sae, hyperparams, metrics, n_training_steps):
        if self.cfg.log_to_wandb and (
            (n_training_steps + 1) % self.cfg.wandb_log_frequency == 0
        ):
            suffix = wandb_log_suffix(self.cfg, hyperparams)
            wandb.log(metrics, step=n_training_steps)

    def checkpoint(self, sae, n_training_tokens, act_freq_scores, n_frac_active_tokens):
        # NOTE fix htis code to not be sae groups anymore
        # path = f"{sae_group.cfg.checkpoint_path}/{n_training_images}_{sae_group.get_name()}.pt"
        self.cfg.save_config(f"{self.cfg.checkpoint_path}/config.json")

        n_training_images = n_training_tokens // self.cfg.context_size
        path = self.cfg.checkpoint_path + f"/n_images_{n_training_images}.pt"
        sae.set_decoder_norm_to_unit_norm()
        sae.save_model(path)

        wandb.log(
            {
                "details/checkpoint_path": path,
            }
        )

        # Save log feature sparsity
        log_feature_sparsity_path = (
            self.cfg.checkpoint_path
            + f"/n_images_{n_training_images}_log_feature_sparsity.pt"
        )
        feature_sparsity = act_freq_scores / n_frac_active_tokens
        log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()
        torch.save(log_feature_sparsity, log_feature_sparsity_path)

        if self.cfg.log_to_wandb:
            hyperparams = sae.cfg
            self.save_to_wandb(sae, hyperparams, path, log_feature_sparsity_path)

    def save_to_wandb(self, sae, hyperparams, path, log_feature_sparsity_path):
        suffix = wandb_log_suffix(sae.cfg, hyperparams)
        name_for_log = re.sub(self.cfg.unique_hash, "_", suffix)
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
                result[field.name] = dataclass_to_dict(value)
            else:
                result[field.name] = value
        return result

    def initalize_wandb(self):
        config_dict = self.dataclass_to_dict(self.cfg)
        run_name = self.cfg.run_name.replace(":", "_")
        wandb_project = self.cfg.wandb_project.replace(":", "_")
        wandb.init(
            project=wandb_project,
            config=config_dict,
            entity=self.cfg.wandb_entity,
            name=run_name,
        )

    def run(self):
        if self.cfg.log_to_wandb:
            self.initalize_wandb()

        (
            act_freq_scores,
            n_forward_passes_since_fired,
            n_frac_active_tokens,
            optimizer,
            scheduler,
        ) = self.initialize_training_variables()
        self.initialize_geometric_medians()

        print("Starting training") if self.cfg.verbose else None

        n_training_steps = 0
        n_training_tokens = 0

        pbar = tqdm(total=self.cfg.total_training_tokens, desc="Training SAE", mininterval=20)
        while n_training_tokens < self.cfg.total_training_tokens:
            layer_acts = self.activations_store.next_batch()

            # init these here to avoid uninitialized vars
            mse_loss = torch.tensor(0.0)
            l1_loss = torch.tensor(0.0)

            (
                loss,
                mse_loss,
                l1_loss,
                l0,
                act_freq_scores,
                n_forward_passes_since_fired,
                n_frac_active_tokens,
            ) = self.train_step(
                sparse_autoencoder=self.sae,
                optimizer=optimizer,
                scheduler=scheduler,
                layer_acts=layer_acts,
                n_training_steps=n_training_steps,
                n_training_tokens=n_training_tokens,
                act_freq_scores=act_freq_scores,
                n_forward_passes_since_fired=n_forward_passes_since_fired,
                n_frac_active_tokens=n_frac_active_tokens,
            )

            # if n_training_steps > 1 and n_training_steps % ((self.cfg.total_training_tokens//self.cfg.train_batch_size)//self.cfg.n_validation_runs) == 0:
            #     self.val(self.sae)


            n_training_steps += 1
            n_training_tokens += self.cfg.train_batch_size

            # if there are still checkpoint thresholds left, check if we need to save a checkpoint
            if (
                len(self.checkpoint_thresholds) > 0
                and n_training_tokens > self.checkpoint_thresholds[0]
            ):
                # Save checkpoint and remove the threshold from the list
                self.checkpoint(
                    self.sae, n_training_tokens, act_freq_scores, n_frac_active_tokens
                )

                if self.cfg.verbose:
                    print(f"Checkpoint saved at {n_training_tokens} tokens")

                self.checkpoint_thresholds.pop(0)

            pbar.update(self.cfg.train_batch_size)

            if l1_loss is None:  # When using top k SAE loss
                l1_loss = torch.tensor(0.0)

            pbar.set_description(
                f"Training SAE: Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, L1 Loss: {l1_loss.item():.4f}, L0: {l0:.4f}", refresh=False
            )

        # Final checkpoint
        if self.cfg.n_checkpoints:
            self.checkpoint(
                self.sae, n_training_tokens, act_freq_scores, n_frac_active_tokens
            )

        if self.cfg.verbose:
            print(f"Final checkpoint saved at {n_training_tokens} tokens")

        pbar.close()

        return self.sae
