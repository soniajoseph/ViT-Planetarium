import torch
import torchvision

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from tqdm import tqdm

import einops
from typing import List

import argparse

import random

import numpy as np
import os
import requests

from plotly.io import write_image
import textwrap

from dataclasses import dataclass
from vit_prisma.sae.config import VisionModelSAERunnerConfig

from vit_prisma.sae.training.activations_store import VisionActivationsStore
# import dataloader
from torch.utils.data import DataLoader

from vit_prisma.utils.data_utils.imagenet_utils import setup_imagenet_paths
from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_transforms_clip, ImageNetValidationDataset
from vit_prisma.models.base_vit import HookedViT

from vit_prisma.sae.sae import SparseAutoencoder

from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_index_to_name

import matplotlib.pyplot as plt

from typing import Any, List, Tuple, Dict

from scipy.stats import gaussian_kde

import json

from dataclasses import dataclass
from functools import partial
from typing import Any, List

import torch
import torch.nn.functional as F

from vit_prisma.models.base_vit import HookedViT
from vit_prisma.sae.sae import SparseAutoencoder


@dataclass
class EvalStats:
    """Evaluation statistics for the quality of an SAE:
        - avg_loss: Average total loss.
        - avg_cos_sim: Average cosine similarity between original and reconstructed
            activations.
        - avg_reconstruction_loss: Average reconstruction loss.
        - avg_zero_abl_loss: Average zero ablation loss.
        - avg_l0: Average L0 sparsity (features activated) per token.
        - avg_l0_cls: Average L0 sparsity for CLS tokens.
        - avg_l0_image: Average L0 sparsity per image.
        - log_frequencies_per_token: Log frequencies of feature activations per token.
        - log_frequencies_per_image: Log frequencies of feature activations per image.
    """

    avg_loss: float
    avg_cos_sim: float
    avg_reconstruction_loss: float
    avg_zero_abl_loss: float
    avg_l0: float
    avg_l0_cls: float
    avg_l0_image: float
    log_frequencies_per_token: np.array
    log_frequencies_per_image: np.array

    def __repr__(self):
        fields = [f"{key}={repr(value)}" for key, value in self.__dict__.items()]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(fields) + "\n)"


@torch.no_grad()
def get_recons_loss(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedViT,
    batch_tokens: torch.Tensor,
    gt_labels: torch.Tensor,
    all_labels: List[str],
    text_embeddings: torch.Tensor,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    # Move model to device if it's not already there
    model = model.to(device)

    # Move all tensors to the same device
    batch_tokens = batch_tokens.to(device)
    gt_labels = gt_labels.to(device)
    text_embeddings = text_embeddings.to(device)

    # Get image embeddings
    image_embeddings, _ = model.run_with_cache(batch_tokens)

    # Calculate similarity scores
    softmax_values, top_k_indices = get_similarity(
        image_embeddings, text_embeddings, device=device
    )

    # Calculate cross-entropy loss
    loss = F.cross_entropy(softmax_values, gt_labels)
    # Safely extract the loss value
    loss_value = loss.item() if torch.isfinite(loss).all() else float("nan")

    head_index = sparse_autoencoder.cfg.hook_point_head_index
    hook_point = sparse_autoencoder.cfg.hook_point

    def standard_replacement_hook(activations: torch.Tensor, hook: Any):
        activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
        return activations

    def head_replacement_hook(activations: torch.Tensor, hook: Any):
        new_activations = sparse_autoencoder.forward(activations[:, :, head_index])[
            0
        ].to(activations.dtype)
        activations[:, :, head_index] = new_activations
        return activations

    replacement_hook = (
        standard_replacement_hook if head_index is None else head_replacement_hook
    )

    recons_image_embeddings = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(hook_point, partial(replacement_hook))],
    )
    recons_softmax_values, _ = get_similarity(
        recons_image_embeddings, text_embeddings, device=device
    )
    recons_loss = F.cross_entropy(recons_softmax_values, gt_labels)

    zero_abl_image_embeddings = model.run_with_hooks(
        batch_tokens, fwd_hooks=[(hook_point, zero_ablate_hook)]
    )
    zero_abl_softmax_values, _ = get_similarity(
        zero_abl_image_embeddings, text_embeddings, device=device
    )
    zero_abl_loss = F.cross_entropy(zero_abl_softmax_values, gt_labels)

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    return score, loss, recons_loss, zero_abl_loss


def get_similarity(image_features, text_features, k=5, device='cuda'):
    image_features = image_features.to(device)
    text_features = text_features.to(device)

    softmax_values = (image_features @ text_features.T).softmax(dim=-1)
    top_k_values, top_k_indices = torch.topk(softmax_values, k, dim=-1)
    return softmax_values, top_k_indices


def zero_ablate_hook(activations: torch.Tensor, hook: Any):
    activations = torch.zeros_like(activations)
    return activations


def get_feature_probability(feature_acts):
    return (feature_acts.abs() > 0).float().flatten(0, 1)


def get_text_labels(name='wordbank'):
    """
    Loads the library of logit labels from a GitHub URL.

    Returns:
    list: A list of string labels.
    """
    if name == 'wordbank':
        url = "https://raw.githubusercontent.com/yossigandelsman/clip_text_span/main/text_descriptions/image_descriptions_general.txt"
        try:
            # Fetch the content from the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Split the content into lines and strip whitespace
            all_labels = [line.strip() for line in response.text.splitlines()]

            print(f"Number of labels loaded: {len(all_labels)}")
            print(f"First 5 labels: {all_labels[:5]}")
            return all_labels

        except requests.RequestException as e:
            print(f"An error occurred while fetching the labels: {e}")
            return []
    elif name == 'imagenet':
        from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_text_labels
        return get_imagenet_text_labels()
    else:
        raise ValueError(f"Invalid label set name: {name}")



def get_text_embeddings(model_name, original_text, batch_size=32):
    from transformers import CLIPProcessor, CLIPModel
    vanilla_model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name, do_rescale=False)

    # Split the text into batches
    text_batches = [original_text[i:i+batch_size] for i in range(0, len(original_text), batch_size)]

    all_embeddings = []

    for batch in text_batches:
        inputs = processor(text=batch, return_tensors='pt', padding=True, truncation=True, max_length=77)
        # inputs = {k: v.to(cfg.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_embeddings = vanilla_model.get_text_features(**inputs)

        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        all_embeddings.append(text_embeddings)

    # Concatenate all batches
    final_embeddings = torch.cat(all_embeddings, dim=0)

    return final_embeddings

def calculate_log_frequencies(total_acts, total_tokens, cfg: VisionModelSAERunnerConfig):
    print("Calculating log frequencies...") if cfg.verbose else None
    # print out all shapes
    print("total_acts shape", total_acts.shape) if cfg.verbose else None
    feature_probs = total_acts / total_tokens
    log_feature_probs = torch.log10(feature_probs)
    return log_feature_probs.cpu().numpy()

def visualize_sparsities(cfg, log_freq_tokens, log_freq_images, conditions, condition_texts, name,
                         sparse_autoencoder):
    # Visualise sparsities for each instance
    hist(
        cfg,
        log_freq_tokens,
        f"{name}_frequency_tokens_histogram",
        show=False,
        title=f"{name} Log Frequency of Features by Token",
        labels={"x": "log<sub>10</sub>(freq)"},
        histnorm="percent",
        template="ggplot2"
    )
    hist(
        cfg,
        log_freq_images,
        f"{name}_frequency_images_histogram",
        show=False,
        title=f"{name} Log Frequency of Features by Image",
        labels={"x": "log<sub>10</sub>(freq)"},
        histnorm="percent",
        template="ggplot2"
    )

    # TODO these conditions need to be tuned to distribution of your data!

    for condition, condition_text in zip(conditions, condition_texts):
        percentage = (torch.count_nonzero(condition) / log_freq_tokens.shape[0]).item() * 100
        if percentage == 0:
            continue
        percentage = int(np.round(percentage))
        rare_encoder_directions = sparse_autoencoder.W_enc[:, condition]
        rare_encoder_directions_normalized = rare_encoder_directions / rare_encoder_directions.norm(dim=0,
                                                                                                    keepdim=True)

        # Compute their pairwise cosine similarities & sample randomly from this N*N matrix of similarities
        cos_sims_rare = (rare_encoder_directions_normalized.T @ rare_encoder_directions_normalized).flatten()
        cos_sims_rare_random_sample = cos_sims_rare[torch.randint(0, cos_sims_rare.shape[0], (10000,))]

        # Plot results
        hist(
            cfg,
            cos_sims_rare_random_sample,
            f"{name}_low_prop_similarity_{condition_text}",
            show=False,
            marginal="box",
            title=f"{name} Cosine similarities of random {condition_text} <br> encoder directions with each other ({percentage}% of features)",
            labels={"x": "Cosine sim"},
            histnorm="percent",
            template="ggplot2",
        )


def get_intervals_for_sparsities(log_freq):
    # Define intervals and conditions
    intervals = [
        (-8, -6),
        (-6, -5),
        (-5, -4),
        (-4, -3),
        (-3, -2),
        (-2, -1),
        (-float('inf'), -8),
        (-1, float('inf'))
    ]

    conditions = [torch.logical_and(log_freq >= lower, log_freq < upper) for lower, upper in intervals]
    condition_texts = [f"TOTAL_logfreq_[{lower},{upper}]" for lower, upper in intervals]

    # Replace infinity with appropriate text for readability
    condition_texts[-2] = condition_texts[-2].replace('-inf', '-∞')
    condition_texts[-1] = condition_texts[-1].replace('inf', '∞')
    return intervals, conditions, condition_texts


def highest_activating_tokens(
        images,
        model,
        sparse_autoencoder,
        W_enc,
        b_enc,
        feature_ids: List[int],
):
    '''
    Returns the indices & values for the highest-activating tokens in the given batch of data.
    '''
    with torch.no_grad():
        # Get the post activations from the clean run
        _, cache = model.run_with_cache(images)

    inp = cache[sparse_autoencoder.cfg.hook_point]
    b, seq_len, _ = inp.shape
    post_reshaped = einops.rearrange(inp, "batch seq d_mlp -> (batch seq) d_mlp")

    # Compute activations
    sae_in = post_reshaped - sparse_autoencoder.b_dec
    acts = einops.einsum(
        sae_in,
        W_enc,
        "... d_in, d_in n -> ... n",
    )
    acts = acts + b_enc
    acts = torch.nn.functional.relu(acts)

    # Reshape acts to (batch, seq, n_features)
    acts_reshaped = einops.rearrange(acts, "(batch seq) n_features -> batch seq n_features", batch=b, seq=seq_len)
    temp_top_indices = {}
    # # The matrix already only contains features due to the selective W_enc you passed in
    # for idx, fid in enumerate(feature_ids): # Iterate through every feature id, and store the corresponding top images/tokens
    #     # Get activations for this feature across all tokens and images
    #     feature_acts = acts_reshaped[:, :, idx].flatten()

    #     # Get top k activating tokens
    #     top_acts_values, top_acts_indices = torch.sort(feature_acts, descending=True)

    #     # Convert flat indices to (image_idx, token_idx) pairs
    #     image_indices = top_acts_indices // seq_len
    #     token_indices = top_acts_indices % seq_len

    #     temp_top_indices[fid] = (list(zip(image_indices.tolist(), token_indices.tolist())), top_acts_values.tolist())

    temp_top_indices = {}
    for idx, fid in enumerate(feature_ids):
        feature_acts = acts_reshaped[:, :, idx].flatten()
        top_acts_values, top_acts_indices = torch.sort(feature_acts, descending=True)

        image_indices = top_acts_indices // seq_len
        token_indices = top_acts_indices % seq_len

        temp_top_indices[fid] = {
            'image_indices': image_indices.tolist(),
            'token_indices': token_indices.tolist(),
            'values': top_acts_values.tolist()
        }
    return temp_top_indices


torch.no_grad()
def get_heatmap(

        image,
        model,
        sparse_autoencoder,
        feature_id,
        device
):
    image = image.to(device)
    _, cache = model.run_with_cache(image.unsqueeze(0))

    post_reshaped = einops.rearrange(cache[sparse_autoencoder.cfg.hook_point], "batch seq d_mlp -> (batch seq) d_mlp")
    # Compute activations (not from a fwd pass, but explicitly, by taking only the feature we want)
    # This code is copied from the first part of the 'forward' method of the AutoEncoder class
    sae_in = post_reshaped - sparse_autoencoder.b_dec  # Remove decoder bias as per Anthropic
    acts = einops.einsum(
        sae_in,
        sparse_autoencoder.W_enc[:, feature_id],
        "x d_in, d_in -> x",
    )
    return acts


def image_patch_heatmap(activation_values, image_size=224, pixel_num=14):
    activation_values = activation_values.detach().cpu().numpy()
    activation_values = activation_values[1:]  # Remove CLS token
    activation_values = activation_values.reshape(pixel_num, pixel_num)

    # Create a heatmap overlay
    heatmap = np.zeros((image_size, image_size))
    patch_size = image_size // pixel_num

    for i in range(pixel_num):
        for j in range(pixel_num):
            heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = activation_values[i, j]

    return heatmap


@torch.no_grad()
def compute_neuron_activations(
    images: torch.Tensor,
    model: torch.nn.Module,
    layer_name: str,
    neuron_indices: List[int],
    top_k: int = 10
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute the highest activating tokens for given neurons in a batch of images.

    Args:
        images: Input images
        model: The main model
        layer_name: Name of the layer to analyze
        neuron_indices: List of neuron indices to analyze
        top_k: Number of top activations to return per neuron

    Returns:
        Dictionary mapping neuron indices to tuples of (top_indices, top_values)
    """
    _, cache = model.run_with_cache(images, names_filter=[layer_name])

    layer_activations = cache[layer_name]

    batch_size, seq_len, n_neurons = layer_activations.shape

    top_activations = {}
    top_k = min(top_k, batch_size)

    for neuron_idx in neuron_indices:
        # Compute mean activation across sequence length
        mean_activations = layer_activations[:, :, neuron_idx].mean(dim=1)
        # Get top-k activations
        top_values, top_indices = mean_activations.topk(top_k)
        top_activations[neuron_idx] = (top_indices, top_values)

    return top_activations


# helper functions
update_layout_set = {"xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor", "showlegend", "xaxis_tickmode", "yaxis_tickmode", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "coloraxis_showscale"}
def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def wrap_text(text, width=60):
    """Wrap text to a specified width."""
    return "<br>".join(textwrap.wrap(text, width=width))


def hist(cfg, tensor, save_name, show=False, renderer=None, **kwargs):
    '''
    '''
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "bargap" not in kwargs_post:
        kwargs_post["bargap"] = 0.1
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])

    histogram_fig = px.histogram(x=to_numpy(tensor), **kwargs_pre)

    # Handle the title separately
    title = kwargs_post.pop('title', '')
    wrapped_title = wrap_text(title, width=60)  # Adjust width as needed

    # Update layout
    histogram_fig.update_layout(
        title={
            'text': wrapped_title,
            'font': {'size': 12},
            'y': 0.98,  # Adjust as needed
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(t=120, b=50, l=50, r=50),  # Increased top margin for multi-line title
        height=650,  # Increased height to accommodate wrapped title
        **kwargs_post
    )

    # save_path = os.path.join(cfg.save_figure_dir, f"{save_name}.png")

    parent_dir = os.path.dirname(cfg.sae_path)
    # Create a new folder path in sae_checkpoints/images with the original name
    save_figure_dir = os.path.join(parent_dir, 'save_fig_dir')
    # Ensure the directory exists
    os.makedirs(save_figure_dir, exist_ok=True)

    # Save the figure as a PNG file
    # Save the figure as PNG and SVG files
    base_path = os.path.join(save_figure_dir, save_name)
    png_path = f"{base_path}.png"
    svg_path = f"{base_path}.svg"

    write_image(histogram_fig, png_path)
    write_image(histogram_fig, svg_path)

    print(f"Histogram saved as PNG: {png_path}")
    print(f"Histogram saved as SVG: {svg_path}")

    if show:
        px.histogram(x=to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post).show(renderer)
    # close
