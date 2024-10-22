import os
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import einops
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_index_to_name
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.evals.eval_utils import (
    EvalStats,
    get_recons_loss,
    get_feature_probability,
    get_text_embeddings,
    get_text_labels,
    calculate_log_frequencies,
    visualize_sparsities,
    get_intervals_for_sparsities,
    highest_activating_tokens,
    get_heatmap,
    image_patch_heatmap,
    compute_neuron_activations,
)
from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.sae.sae_utils import wandb_log_suffix
from vit_prisma.utils.constants import EvaluationContext


@contextmanager
def eval_mode(sae: nn.Module):
    """Context manager to temporarily switch to evaluation mode."""
    is_train = sae.training
    try:
        sae.eval()
        yield sae
    finally:
        if is_train:
            sae.train()


class Evaluator:
    def __init__(
        self,
        model: HookedViT,
        data: torch.utils.data.Dataset,
        cfg: VisionModelSAERunnerConfig,
        visualize_data: torch.utils.data.Dataset,
    ):
        """A class that holds various evaluations for SAEs trained on a given
        model and dataset.
        """
        self.model = model
        self.data = data
        self.visualize_data = visualize_data
        self.cfg = cfg
        self._evaluation_cfg = None
        self.neuron_indices = list(range(self.cfg.d_in))

    @property
    def evaluation_cfg(self):
        return self._evaluation_cfg

    @property
    def dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.evaluation_cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

    @evaluation_cfg.setter
    def evaluation_cfg(self, context: EvaluationContext):
        if context == EvaluationContext.TRAINING:
            self._evaluation_cfg = self.cfg.training_eval
        elif context == EvaluationContext.POST_TRAINING:
            self._evaluation_cfg = self.cfg.post_training_eval
        else:
            raise ValueError(
                "Invalid evaluation context (options are 'training' and "
                "'post-training')"
            )

    def evaluate(
        self,
        sae: SparseAutoencoder,
        context: EvaluationContext = EvaluationContext.TRAINING,
    ):
        """The type of evaluation is determined by the `context` parameter. All
        evaluations functions specified in the `EvalConfig` are run.
        """
        self.evaluation_cfg = context
        print(f"Performing {context.value} mode evaluation on SAE")

        with eval_mode(sae):
            stats = self.process_dataset(sae)
            self.save_stats(sae, stats)

            for func_name in self.evaluation_cfg.evaluation_functions:
                print(f"Running the evaluation: {func_name.value}")
                eval_func = getattr(self, func_name.value, None)
                if eval_func:
                    eval_func(sae, stats)
                else:
                    print(
                        f"Warning: Evaluation function '{func_name.value}' not implemented"
                    )

    def process_dataset(self, sae):
        """This function evaluates the performance of a sparse autoencoder on a dataset,
        computing statistics such as L0 sparsity, reconstruction quality, and loss
        metrics.

        NB. Currently the function assumes the use of ImageNet dataset and a CLIP model.
        """
        all_l0 = []
        all_l0_cls = []
        all_l0_image = []

        total_loss = 0
        total_reconstruction_loss = 0
        total_zero_abl_loss = 0
        total_samples = 0
        all_cosine_similarity = []

        all_labels = get_text_labels("imagenet")
        text_embeddings = get_text_embeddings(self.cfg.model_name, all_labels)

        total_acts = None
        total_tokens = 0

        with torch.no_grad():
            for batch_tokens, gt_labels, indices in self.dataloader:
                batch_tokens = batch_tokens.to(self.cfg.device)
                batch_size = batch_tokens.shape[0]

                total_samples += batch_size

                _, cache = self.model.run_with_cache(
                    batch_tokens, names_filter=sae.cfg.hook_point
                )
                hook_point_activation = cache[sae.cfg.hook_point].to(self.cfg.device)

                sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sae(
                    hook_point_activation
                )

                # Calculate feature probability
                sae_activations = get_feature_probability(feature_acts)
                if total_acts is None:
                    total_acts = sae_activations.sum(0)
                else:
                    total_acts += sae_activations.sum(0)

                total_tokens += sae_activations.shape[0]

                # Get L0 stats per token
                l0 = (feature_acts[:, 1:, :] > 0).float().sum(-1).detach()
                all_l0.extend(l0.mean(dim=1).cpu().numpy())
                l0_cls = (feature_acts[:, 0, :] > 0).float().sum(-1).detach()
                all_l0_cls.extend(l0_cls.flatten().cpu().numpy())

                # Get L0 stats per image
                l0 = (feature_acts > 0).float().sum(-1).detach()
                image_l0 = l0.sum(dim=1)  # Sum across all tokens
                all_l0_image.extend(image_l0.cpu().numpy())

                # Calculate cosine similarity between original activations and sae output
                cos_sim = (
                    torch.cosine_similarity(
                        einops.rearrange(
                            hook_point_activation,
                            "batch seq d_mlp -> (batch seq) d_mlp",
                        ),
                        einops.rearrange(
                            sae_out, "batch seq d_mlp -> (batch seq) d_mlp"
                        ),
                        dim=0,
                    )
                    .mean(-1)
                    .tolist()
                )
                all_cosine_similarity.append(cos_sim)

                # Calculate substitution loss
                score, loss, recons_loss, zero_abl_loss = get_recons_loss(
                    sae,
                    self.model,
                    batch_tokens,
                    gt_labels,
                    all_labels,
                    text_embeddings,
                    device=self.cfg.device,
                )

                total_loss += loss.item()
                total_reconstruction_loss += recons_loss.item()
                total_zero_abl_loss += zero_abl_loss.item()

                if total_samples >= self.evaluation_cfg.max_evaluation_images:
                    break

        # Calculate average metrics
        avg_loss = total_loss / total_samples
        avg_reconstruction_loss = total_reconstruction_loss / total_samples
        avg_zero_abl_loss = total_zero_abl_loss / total_samples

        avg_l0 = np.mean(all_l0)
        avg_l0_cls = np.mean(all_l0_cls)
        avg_l0_image = np.mean(all_l0_image)

        avg_cos_sim = np.mean(all_cosine_similarity)
        log_frequencies_per_token = calculate_log_frequencies(
            total_acts, total_tokens, self.cfg
        )
        log_frequencies_per_image = calculate_log_frequencies(
            total_acts, total_samples, self.cfg
        )

        return EvalStats(
            avg_loss,
            avg_cos_sim,
            avg_reconstruction_loss,
            avg_zero_abl_loss,
            avg_l0,
            avg_l0_cls,
            avg_l0_image,
            log_frequencies_per_token,
            log_frequencies_per_image,
        )

    def save_stats(self, sae: SparseAutoencoder, stats: EvalStats):
        """Store the evaluation stats to wandb."""

        print(f"Logging evaluation stats to wandb") if self.cfg.verbose else None
        suffix = wandb_log_suffix(sae.cfg, self.cfg)

        metrics = {
            f"validation/{field.name}{suffix}": getattr(stats, field.name)
            for field in fields(stats)
        }
        if self.cfg.log_to_wandb:
            wandb.log(metrics)

    def evaluate_sae_features(self, sae, stats: EvalStats):
        """Plot and analyze log frequencies of feature activations for an SAE.

        This function performs several tasks:
            1. Visualises sparsity patterns across different activation frequency ranges.
            2. Samples representative features from different activation frequency
                intervals of interest.
            3. Identifies top-activating images for sampled features.
            4. Generates and saves visualizations of these top-activating images with
                activation heatmaps.

        NB:
            - The function assumes the use of an ImageNet dataset.
            - The function saves output images in a directory structure based on the
                SAE checkpoint path.
        """

        print("Plotting log frequencies...")
        log_freq_tokens = torch.Tensor(stats.log_frequencies_per_token)
        log_freq_images = torch.Tensor(stats.log_frequencies_per_image)
        intervals, conditions, conditions_texts = get_intervals_for_sparsities(
            log_freq_tokens
        )
        visualize_sparsities(
            self.cfg,
            log_freq_tokens,
            log_freq_images,
            conditions,
            conditions_texts,
            "TOTAL",
            sae,
        )

        print("Sampling features from pre-specified intervals...")
        # get random features from different bins
        interesting_features_indices = []
        interesting_features_values = []
        interesting_features_category = []
        # number_features_per = 10
        for condition, condition_text in zip(conditions, conditions_texts):
            potential_indices = torch.nonzero(condition, as_tuple=True)[0]

            # Shuffle these indices and select a subset
            sampled_indices = potential_indices[
                torch.randperm(len(potential_indices))[
                    : self.evaluation_cfg.samples_per_bin
                ]
            ]

            values = log_freq_tokens[sampled_indices]

            interesting_features_indices = (
                interesting_features_indices + sampled_indices.tolist()
            )
            interesting_features_values = interesting_features_values + values.tolist()
            interesting_features_category = interesting_features_category + [
                f"{condition_text}"
            ] * len(sampled_indices)

        print(set(interesting_features_category))

        print("Running through dataset to get top images per feature...")
        max_indices = {i: None for i in interesting_features_indices}
        max_values = {i: None for i in interesting_features_indices}
        b_enc = sae.b_enc[interesting_features_indices]
        W_enc = sae.W_enc[:, interesting_features_indices]

        for batch_idx, (total_images, total_labels, total_indices) in enumerate(
            self.dataloader
        ):
            total_images = total_images.to(self.cfg.device)
            total_indices = total_indices.to(self.cfg.device)

            new_top_info = highest_activating_tokens(
                total_images,
                self.model,
                sae,
                W_enc,
                b_enc,
                interesting_features_indices,
            )  # Return all

            for feature_id in interesting_features_indices:
                self._update_top_activations(
                    feature_id,
                    new_top_info[feature_id],
                    total_indices,
                    max_indices,
                    max_values,
                )

            if (
                batch_idx * self.evaluation_cfg.batch_size
                >= self.evaluation_cfg.max_evaluation_images
            ):
                break

        top_per_feature = {
            i: (max_values[i].detach().cpu(), max_indices[i].detach().cpu())
            for i in interesting_features_indices
        }
        self._plot_top_activating_images(
            sae,
            top_per_feature,
            interesting_features_category,
            interesting_features_indices,
        )

    def find_top_activations_for_neurons(
        self,
        sae: SparseAutoencoder,
        stats: EvalStats,
        top_k: int = 16,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        max_samples = self.evaluation_cfg.max_evaluation_images

        top_activations = {i: (None, None) for i in self.neuron_indices}

        processed_samples = 0
        for batch_images, _, batch_indices in self.dataloader:
            batch_images = batch_images.to(self.cfg.device)
            batch_indices = batch_indices.to(self.cfg.device)
            batch_size = batch_images.shape[0]

            batch_activations = compute_neuron_activations(
                batch_images,
                self.model,
                self.cfg.hook_point,
                self.neuron_indices,
                top_k,
            )

            for neuron_idx in self.neuron_indices:
                new_indices, new_values = batch_activations[neuron_idx]
                new_indices = batch_indices[new_indices]

                if top_activations[neuron_idx][0] is None:
                    top_activations[neuron_idx] = (new_values, new_indices)
                else:
                    combined_values = torch.cat(
                        (top_activations[neuron_idx][0], new_values)
                    )
                    combined_indices = torch.cat(
                        (top_activations[neuron_idx][1], new_indices)
                    )
                    _, top_k_indices = torch.topk(combined_values, top_k)
                    top_activations[neuron_idx] = (
                        combined_values[top_k_indices],
                        combined_indices[top_k_indices],
                    )

            processed_samples += batch_size
            if processed_samples >= max_samples:
                break

        top_activations_per_neuron = {
            i: (values.detach().cpu(), indices.detach().cpu())
            for i, (values, indices) in top_activations.items()
        }
        self._visualize_top_activations(sae, top_activations_per_neuron)

    def _visualize_top_activations(
        self,
        sae: SparseAutoencoder,
        top_activations_per_neuron: int,
    ):
        ind_to_name = get_imagenet_index_to_name()
        parent_dir = Path(self.cfg.sae_path).parent
        folder = str(parent_dir / "max_images")
        os.makedirs(folder, exist_ok=True)
        print(f"Saving top activations images to {folder}")

        for neuron_idx in tqdm(self.neuron_indices, total=len(self.neuron_indices)):
            max_vals, max_inds = top_activations_per_neuron[neuron_idx]
            images = []
            model_images = []
            gt_labels = []

            for bid, v in zip(max_inds, max_vals):
                image, label, image_ind = self.visualize_data[bid]
                assert image_ind.item() == bid
                images.append(image)

                model_image, _, _ = self.data[bid]
                model_images.append(model_image)
                gt_labels.append(ind_to_name[str(label)][1])

            grid_size = int(np.ceil(np.sqrt(len(images))))
            fig, axs = plt.subplots(
                int(np.ceil(len(images) / grid_size)), grid_size, figsize=(15, 15)
            )
            name = f"Layer: {self.cfg.hook_point}, Neuron: {neuron_idx}"
            fig.suptitle(name)

            for ax in axs.flatten():
                ax.axis("off")

            complete_bid = []

            for i, (image_tensor, label, val, bid, model_img) in enumerate(
                zip(images, gt_labels, max_vals, max_inds, model_images)
            ):
                image_tensor, model_img = image_tensor.to(
                    self.cfg.device
                ), model_img.to(self.cfg.device)

                if bid in complete_bid:
                    continue
                complete_bid.append(bid)

                row = i // grid_size
                col = i % grid_size
                heatmap = get_heatmap(model_img, self.model, sae, neuron_idx)
                heatmap = image_patch_heatmap(
                    heatmap,
                    image_size=self.cfg.image_size,
                    pixel_num=self.cfg.num_patch,
                )

                display = image_tensor.numpy().transpose(1, 2, 0)

                axs[row, col].imshow(display)
                axs[row, col].imshow(
                    heatmap, cmap="viridis", alpha=0.3
                )  # Overlaying the heatmap
                axs[row, col].set_title(f"{label} {val.item():0.03f}")
                axs[row, col].axis("off")

            plt.tight_layout()

            plt.savefig(os.path.join(folder, f"neuron_{neuron_idx}.png"))
            plt.close()

    def _update_top_activations(
        self,
        feature_id: int,
        feature_data: Dict,
        total_indices: torch.Tensor,
        max_indices: Dict[int, torch.Tensor],
        max_values: Dict[int, torch.Tensor],
    ):
        batch_image_indices = torch.tensor(feature_data["image_indices"])
        token_activation_values = torch.tensor(
            feature_data["values"], device=self.cfg.device
        )
        global_image_indices = total_indices[batch_image_indices]  # Get global indices

        # Get unique image indices and their highest activation values
        unique_image_indices, unique_indices = torch.unique(
            global_image_indices, return_inverse=True
        )
        unique_activation_values = torch.zeros_like(
            unique_image_indices, dtype=torch.float, device=self.cfg.device
        )
        unique_activation_values.index_reduce_(
            0, unique_indices, token_activation_values, "amax"
        )

        if max_indices[feature_id] is None:
            max_indices[feature_id] = unique_image_indices
            max_values[feature_id] = unique_activation_values
        else:
            # Concatenate with existing data
            all_indices = torch.cat((max_indices[feature_id], unique_image_indices))
            all_values = torch.cat((max_values[feature_id], unique_activation_values))

            # Get unique indices again (in case of overlap between batches)
            unique_all_indices, unique_all_idx = torch.unique(
                all_indices, return_inverse=True
            )
            unique_all_values = torch.zeros_like(unique_all_indices, dtype=torch.float)
            unique_all_values.index_reduce_(0, unique_all_idx, all_values, "amax")

            # Select top k
            if len(unique_all_indices) > self.evaluation_cfg.max_images_per_feature:
                _, top_k_idx = torch.topk(
                    unique_all_values,
                    k=self.evaluation_cfg.max_images_per_feature,
                )
                max_indices[feature_id] = unique_all_indices[top_k_idx]
                max_values[feature_id] = unique_all_values[top_k_idx]
            else:
                max_indices[feature_id] = unique_all_indices
                max_values[feature_id] = unique_all_values

    def _plot_top_activating_images(
        self,
        sae: SparseAutoencoder,
        top_per_feature,
        interesting_features_category,
        interesting_features_values,
    ):
        # Create a new folder path in sae_checkpoints/images with the original name
        output_dir = Path(self.cfg.sae_path).parent / "max_images"
        os.makedirs(output_dir, exist_ok=True)
        print(
            f"Saving the images that represent the feature density of interest in the"
            f" directory {output_dir}"
        )

        ind_to_name = get_imagenet_index_to_name()

        for feature_ids, cat, logfreq in zip(
            top_per_feature.keys(),
            interesting_features_category,
            interesting_features_values,
        ):
            max_vals, max_inds = top_per_feature[feature_ids]
            images = []
            model_images = []
            gt_labels = []
            unique_bids = set()
            for bid, v in zip(max_inds, max_vals):
                if len(unique_bids) >= self.evaluation_cfg.max_images_per_feature:
                    break
                if bid not in unique_bids:
                    image, label, image_ind = self.visualize_data[bid]
                    images.append(image)
                    model_img, _, _ = self.data[bid]
                    model_images.append(model_img)
                    gt_labels.append(ind_to_name[str(label)][1])
                    unique_bids.add(bid)

            grid_size = int(np.ceil(np.sqrt(len(images))))
            fig, axs = plt.subplots(
                int(np.ceil(len(images) / grid_size)), grid_size, figsize=(15, 15)
            )
            name = f"Category: {cat},  Feature: {feature_ids}"
            fig.suptitle(name)  # , y=0.95)
            for ax in axs.flatten():
                ax.axis("off")
            complete_bid = []

            for i, (image_tensor, label, val, bid, model_img) in enumerate(
                zip(images, gt_labels, max_vals, max_inds, model_images)
            ):
                image_tensor, model_img = image_tensor.to(
                    self.cfg.device
                ), model_img.to(self.cfg.device)

                if bid in complete_bid:
                    continue
                complete_bid.append(bid)

                row = i // grid_size
                col = i % grid_size

                heatmap = get_heatmap(model_img, self.model, sae, feature_ids)
                heatmap = image_patch_heatmap(
                    heatmap,
                    image_size=self.cfg.image_size,
                    pixel_num=self.cfg.num_patch,
                )
                display = image_tensor.numpy().transpose(1, 2, 0)

                has_zero = False

                axs[row, col].imshow(display)
                axs[row, col].imshow(
                    heatmap, cmap="viridis", alpha=0.3
                )  # Overlaying the heatmap
                axs[row, col].set_title(
                    f"{label} {val.item():0.06f} {'class token!' if has_zero else ''}"
                )
                axs[row, col].axis("off")

            plt.tight_layout()

            folder = os.path.join(output_dir, f"{cat}")
            os.makedirs(folder, exist_ok=True)
            plt.savefig(
                os.path.join(
                    folder, f"neglogfreq_{-logfreq}_feature_id:{feature_ids}.png"
                )
            )
            # save svg
            plt.savefig(
                os.path.join(
                    folder, f"neglogfreq_{-logfreq}_feature_id:{feature_ids}.svg"
                )
            )
            plt.close()
