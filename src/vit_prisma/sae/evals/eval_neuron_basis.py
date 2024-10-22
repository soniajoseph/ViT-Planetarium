from typing import List, Dict, Tuple
import torch
from tqdm import tqdm

import torch
import einops
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from vit_prisma.models.base_vit import HookedViT
 
# load dataset
from vit_prisma.utils.data_utils.imagenet.imagenet_utils import setup_imagenet_paths
from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_transforms_clip, ImageNetValidationDataset

from dataclasses import dataclass
from vit_prisma.sae.config import VisionModelSAERunnerConfig

import torchvision

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

def find_top_activations_for_neurons(
    val_dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    cfg: object,
    layer_name: str,
    neuron_indices: List[int],
    top_k: int = 16,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Find the top activations for specific neurons across the validation dataset.

    Args:
        val_dataloader: Validation data loader
        model: The main model
        cfg: Configuration object
        layer_name: Name of the layer to analyze
        neuron_indices: Indices of neurons to analyze
        top_k: Number of top activations to return per neuron

    Returns:
        Dictionary mapping neuron indices to tuples of (top_values, top_indices)
    """
    max_samples = cfg.eval_max

    top_activations = {i: (None, None) for i in neuron_indices}

    processed_samples = 0
    for batch_images, _, batch_indices in tqdm(val_dataloader, total=max_samples // cfg.batch_size):
        batch_images = batch_images.to(cfg.device)
        batch_indices = batch_indices.to(cfg.device)
        batch_size = batch_images.shape[0]

        batch_activations = compute_neuron_activations(
            batch_images, model, layer_name, neuron_indices, top_k
        )

        for neuron_idx in neuron_indices:
            new_indices, new_values = batch_activations[neuron_idx]
            new_indices = batch_indices[new_indices]
            
            if top_activations[neuron_idx][0] is None:
                top_activations[neuron_idx] = (new_values, new_indices)
            else:
                combined_values = torch.cat((top_activations[neuron_idx][0], new_values))
                combined_indices = torch.cat((top_activations[neuron_idx][1], new_indices))
                _, top_k_indices = torch.topk(combined_values, top_k)
                top_activations[neuron_idx] = (combined_values[top_k_indices], combined_indices[top_k_indices])

        processed_samples += batch_size
        if processed_samples >= max_samples:
            break

    return {i: (values.detach().cpu(), indices.detach().cpu()) 
            for i, (values, indices) in top_activations.items()}

@torch.no_grad()
def get_heatmap(
    image,
    model,
    layer_name,
    neuron_idx,
):
    image = image.to(cfg.device)
    _, cache = model.run_with_cache(image.unsqueeze(0), names_filter=[layer_name])
    
    layer_activations = cache[layer_name]
    neuron_activations = layer_activations[0, :, neuron_idx]
    
    return neuron_activations

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

def visualize_top_activations(
    model,
    val_data,
    val_data_visualize,
    top_activations_per_neuron,
    layer_name,
    neuron_indices,
    ind_to_name,
    cfg
):
    print("Saving to ", cfg.max_image_output_folder)
    for neuron_idx in tqdm(neuron_indices, total=len(neuron_indices)):
        max_vals, max_inds = top_activations_per_neuron[neuron_idx]
        images = []
        model_images = []
        gt_labels = []
        
        for bid, v in zip(max_inds, max_vals):
            image, label, image_ind = val_data_visualize[bid]
            assert image_ind.item() == bid
            images.append(image)
            
            model_image, _, _ = val_data[bid]
            model_images.append(model_image)
            gt_labels.append(ind_to_name[str(label)][1])
        
        grid_size = int(np.ceil(np.sqrt(len(images))))
        fig, axs = plt.subplots(int(np.ceil(len(images)/grid_size)), grid_size, figsize=(15, 15))
        name = f"Layer: {layer_name}, Neuron: {neuron_idx}"
        fig.suptitle(name)
        
        for ax in axs.flatten():
            ax.axis('off')
        
        complete_bid = []
        
        for i, (image_tensor, label, val, bid, model_img) in enumerate(zip(images, gt_labels, max_vals, max_inds, model_images)):
            if bid in complete_bid:
                continue
            complete_bid.append(bid)
            
            row = i // grid_size
            col = i % grid_size
            heatmap = get_heatmap(model_img, model, layer_name, neuron_idx)
            heatmap = image_patch_heatmap(heatmap, pixel_num=224//cfg.patch_size)
            
            display = image_tensor.numpy().transpose(1, 2, 0)
            
            axs[row, col].imshow(display)
            axs[row, col].imshow(heatmap, cmap='viridis', alpha=0.3)  # Overlaying the heatmap
            axs[row, col].set_title(f"{label} {val.item():0.03f}")
            axs[row, col].axis('off')
        
        plt.tight_layout()
        
        folder = os.path.join(cfg.max_image_output_folder, f"{layer_name}")
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, f"neuron_{neuron_idx}.png"))
        plt.close()


@dataclass
class EvalConfig(VisionModelSAERunnerConfig):
    # sae_path: str = '/network/scratch/s/sonia.joseph/sae_checkpoints/1f89d99e-wkcn-TinyCLIP-ViT-40M-32-Text-19M-LAION400M-expansion-16/n_images_520028.pt'
    model_name: str = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"
    model_type: str =  "clip"
    patch_size: str = 32

    save_directory = "/network/scratch/s/sonia.joseph/sae_checkpoints/neuron_basis"

    dataset_path = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets"
    dataset_train_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train"
    dataset_val_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val"

    verbose: bool = True

    device: bool = 'cuda'

    eval_max: int = 50_000 # 50_000
    batch_size: int = 32

    # make the max image output folder a subfolder of the sae path


    @property
    def max_image_output_folder(self) -> str:
        # Use the model name in the output folder path
        model_name = self.model_name.replace('/', '_')  # Replace '/' with '_' for valid directory name
        output_folder = os.path.join(self.save_directory, model_name)
        os.makedirs(output_folder, exist_ok=True)
        return output_folder


####
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hook_point_layer', type=int, default=None)
    args = parser.parse_args()

    cfg = EvalConfig()

    if args.hook_point_layer:
        cfg.hook_point_layer = args.hook_point_layer


     # Load npy file for neuron indices
    indices_path = '/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/tinyclip_neuron_indices_mlp_out.npy'
    neuron_indices_dictionary = np.load(indices_path, allow_pickle=True).item()
    neuron_indices = neuron_indices_dictionary[cfg.hook_point_layer]
    print("Neuron indices are length: ", len(neuron_indices), "for layer: ", cfg.hook_point_layer)

    model = HookedViT.from_pretrained(cfg.model_name, is_timm=False, is_clip=True).to(cfg.device)

    if cfg.model_type == 'clip':
        data_transforms = get_imagenet_transforms_clip(cfg.model_name)
    else:
        raise ValueError("Invalid model type")
    imagenet_paths = setup_imagenet_paths(cfg.dataset_path)
    train_data = torchvision.datasets.ImageFolder(cfg.dataset_train_path, transform=data_transforms)
    val_data = ImageNetValidationDataset(cfg.dataset_val_path, 
                                    imagenet_paths['label_strings'], 
                                    imagenet_paths['val_labels'], 
                                    data_transforms,
                                    return_index=True,
    )
    val_data_visualize = ImageNetValidationDataset(cfg.dataset_val_path, 
                                    imagenet_paths['label_strings'], 
                                    imagenet_paths['val_labels'],
                                    torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),]), return_index=True)

    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_index_to_name
    ind_to_name = get_imagenet_index_to_name()

    print(f"Validation data length: {len(val_data)}") if cfg.verbose else None

    print("Finding top activations for neurons")
    top_activations_per_neuron = find_top_activations_for_neurons(
        val_dataloader,
        model,
        cfg,
        cfg.hook_point,
        neuron_indices,
        top_k=20,
    )

    print("Visualizing top activations")
    visualize_top_activations(
        model,
        val_data,
        val_data_visualize,
        top_activations_per_neuron,
        cfg.hook_point,
        neuron_indices,
        ind_to_name,
        cfg
    )