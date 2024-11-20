from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.sae.evals.evals import (get_substitution_loss, load_model, load_sae, get_text_labels, get_text_embeddings_openclip, 
                                        get_feature_probability, calculate_log_frequencies, get_intervals_for_sparsities, get_heatmap, highest_activating_tokens,
                                        image_patch_heatmap)
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.utils.data_utils.imagenet.imagenet_utils import setup_imagenet_paths
from vit_prisma.dataloaders.imagenet_index import imagenet_index
from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_index_to_name
from vit_prisma.transforms.open_clip_transforms import get_clip_val_transforms

import open_clip

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from PIL import Image
import os
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

import einops

import sys
import torch
from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.evals.evals import ImageNetValidationDataset
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download
import numpy as np

import h5py


def load_and_test_sae(repo_id="prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-0.0001", checkpoint="n_images_2600058.pt"):
    """
    Load and test SAE from HuggingFace
    """
    print(f"Loading model from {repo_id}...")

    # Download config.json and  get path 

    sae_path = hf_hub_download(repo_id, checkpoint)
    config_path = hf_hub_download(repo_id, 'config.json')

    sae = SparseAutoencoder.load_from_pretrained(sae_path)

    print(sae)

    print(type(sae.cfg))

    # Move to available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = sae.to(device)
    print(f"Using device: {device}")

    return sae

class SAEEval():

    def __init__(self, sae: SparseAutoencoder):
        
        self.sae = sae
        self.model = load_model(sae.cfg)

        self.lr = sae.cfg.lr
        self.l1_coeff = sae.cfg.l1_coefficient

        imagenet_paths = setup_imagenet_paths(self.sae.cfg.dataset_path)

        data_transforms = get_clip_val_transforms()

        self.validation_dataset = ImageNetValidationDataset(
            self.sae.cfg.dataset_val_path,
            imagenet_paths['label_strings'], 
            imagenet_paths['val_labels'], 
            data_transforms,
            return_index=True,
        )

        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=256, shuffle=True, num_workers=4)
    


    def save_image_grid(self, images, labels, activation_values, image_indices, ind_to_name, folder, feature_idx):

        num_images = len(images)
        grid_cols = 5
        grid_rows = 4
        
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols*3, grid_rows*3))
        axes = axes.flatten()
        
        for idx, (image_tensor, label_i, activation_value, image_idx) in enumerate(zip(images, labels, activation_values, image_indices)):
            ax = axes[idx]

            model_image = image_tensor.to(self.sae.device).unsqueeze(0)  
            heatmap_activations = get_heatmap(model_image.squeeze(0), self.model, self.sae, feature_idx, self.sae.device)
            heatmap = image_patch_heatmap(heatmap_activations, self.sae.cfg)
            display = image_tensor.cpu().numpy().transpose(1, 2, 0)
            
            display_min = display.min()
            display_max = display.max()
            if display_max > display_min: 
                display = (display - display_min) / (display_max - display_min)
            else:
                display = display - display_min  

            ax.imshow(display)
            ax.imshow(heatmap, cmap='viridis', alpha=0.3)
            label_name = ind_to_name.get(str(label_i), ["unknown"])[1]
            ax.set_title(f"{label_name}\nActivation: {activation_value:.4f}")
            ax.axis('off')
        
        for idx in range(num_images, grid_rows * grid_cols):
            axes[idx].axis('off')
        
        plt.tight_layout()
        grid_image_filename = os.path.join(folder, "grid_image.png")
        plt.savefig(grid_image_filename, dpi=300)
        plt.close()
        print(f"Saved grid image at {grid_image_filename}")


    def run_eval(self, path: str):

        all_l0 = []
        all_l0_cls = []

        # image level l0
        all_l0_image = []

        total_loss = 0
        total_score = 0
        total_reconstruction_loss = 0
        total_zero_abl_loss = 0
        total_samples = 0
        all_cosine_similarity = []
        all_recons_cosine_similarity = []

        self.model.eval()
        self.sae.eval()

        all_labels = get_text_labels('imagenet')

        num_imagenet_classes = 1000
        batch_label_names = [imagenet_index[str(int(label))][1] for label in range(num_imagenet_classes)]

        og_model, _, preproc = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K')
        tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K')


        text_embeddings = get_text_embeddings_openclip(og_model, preproc, tokenizer, batch_label_names)

        total_acts = None
        total_tokens = 0
        total_images = 0
        alive_features = None

        with torch.no_grad():

            num_features = self.sae.W_enc.shape[1]
            random_feature_indices = np.random.choice(num_features, size=30, replace=False)

            top_mean_activations_per_feature = {i: [] for i in random_feature_indices}

            total_patches = 0
            feature_activation_counts = torch.zeros(num_features, device='cuda')
            feature_activation_sums = torch.zeros(num_features, device='cuda')
            feature_activation_squares = torch.zeros(num_features, device='cuda')
            

            for batch_tokens, gt_labels, indices, labels in tqdm(self.validation_dataloader):
                batch_tokens = batch_tokens.to(self.sae.device)
                batch_size = batch_tokens.shape[0]
                # batch shape
                total_samples += batch_size
                _, cache = self.model.run_with_cache(batch_tokens, names_filter=self.sae.cfg.hook_point)
                hook_point_activation = cache[self.sae.cfg.hook_point].to(self.sae.device)
                
                sae_out, feature_acts, loss, mse_loss, l1_loss, _, aux_loss = self.sae(hook_point_activation)

                batch_size, seq_len, _ = feature_acts.shape
                total_patches += batch_size * seq_len

                feature_acts_flat = feature_acts.reshape(-1, num_features)
                feature_active = (feature_acts_flat > 0).float()
                feature_activation_counts += feature_active.sum(dim=0)
                feature_activation_sums += feature_acts_flat.sum(dim=0)
                feature_activation_squares += (feature_acts_flat ** 2).sum(dim=0)

                mean_activation_per_image = feature_acts.mean(dim=1)
                mean_activation_per_image_selected_features = mean_activation_per_image[:, random_feature_indices]

                for i, feature_idx in enumerate(random_feature_indices):
                    
                    mean_activations = mean_activation_per_image_selected_features[:, i].cpu().numpy()

                    image_indices = indices.cpu().numpy()
                    
                    mean_current_list = top_mean_activations_per_feature[feature_idx]

                    mean_current_list.extend(zip(mean_activations.tolist(), image_indices.tolist()))

                    mean_current_list.sort(key=lambda x: x[0], reverse=True)
                    if len(mean_current_list) > 20:
                        mean_current_list = mean_current_list[:20]

                    top_mean_activations_per_feature[feature_idx] = mean_current_list

                sae_activations = get_feature_probability(feature_acts)
                if total_acts is None:
                    total_acts = sae_activations.sum(0)
                else:
                    total_acts += sae_activations.sum(0)
                
                total_tokens += sae_activations.shape[0]
                total_images += batch_size

                l0 = (feature_acts[:, 1:, :] > 0).float().sum(-1).detach()
                if alive_features is None:
                    alive_features = torch.zeros(feature_acts.shape[-1], dtype=torch.bool).cpu()
                    alive_features |= ((feature_acts[:, 1:, :] > 0).any(dim=(0, 1)).cpu().bool())
                else:
                    alive_features |= ((feature_acts[:, 1:, :] > 0).any(dim=(0, 1)).cpu().bool())
                all_l0.extend(l0.mean(dim=1).cpu().numpy())
                l0_cls = (feature_acts[:, 0, :] > 0).float().sum(-1).detach()
                all_l0_cls.extend(l0_cls.flatten().cpu().numpy())

                l0 = (feature_acts > 0).float().sum(-1).detach()
                image_l0 = l0.sum(dim=1)  
                all_l0_image.extend(image_l0.cpu().numpy())

                cos_sim = torch.cosine_similarity(einops.rearrange(hook_point_activation, "batch seq d_mlp -> (batch seq) d_mlp"),
                                                                                einops.rearrange(sae_out, "batch seq d_mlp -> (batch seq) d_mlp"),
                                                                                    dim=0).mean(-1).tolist()
                all_cosine_similarity.append(cos_sim)

                score, loss, recons_loss, zero_abl_loss, recons_cosine_sim = get_substitution_loss(self.sae, self.model, batch_tokens, gt_labels, 
                                                                        text_embeddings, device=self.sae.device)
                total_loss += loss.item()
                total_score += score.item()
                total_reconstruction_loss += recons_loss.item()
                total_zero_abl_loss += zero_abl_loss.item()
                all_recons_cosine_similarity.extend(recons_cosine_sim)

        feature_activation_counts.to('cpu')
        feature_activation_sums.to('cpu')
        feature_activation_squares.to('cpu')

        avg_loss = total_loss / len(self.validation_dataloader)
        avg_reconstruction_loss = total_reconstruction_loss / len(self.validation_dataloader)
        avg_zero_abl_loss = total_zero_abl_loss / len(self.validation_dataloader)
        avg_score = total_score / len(self.validation_dataloader)

        ce_recovered =  ((avg_zero_abl_loss - avg_reconstruction_loss) / (avg_zero_abl_loss - avg_loss)) * 100
        
        avg_l0 = np.mean(all_l0)
        avg_l0_cls = np.mean(all_l0_cls)
        avg_l0_image = np.mean(all_l0_image)

        avg_cos_sim = np.mean(all_cosine_similarity)
        total_recons_cosine_similarity = np.mean(all_recons_cosine_similarity)

        metrics = {}
        metrics['avg_l0'] = avg_l0.astype(float)
        metrics['avg_cls_l0'] = avg_l0_cls.astype(float)
        metrics['avg_image_l0'] = avg_l0_image.astype(float)
        metrics['avg_cosine_similarity'] = avg_cos_sim.astype(float)
        metrics['avg_recons_cosine_similarity'] = total_recons_cosine_similarity.astype(float)
        metrics['avg_CE'] = avg_loss
        metrics['avg_recons_CE'] = avg_reconstruction_loss
        metrics['avg_zero_abl_CE'] = avg_zero_abl_loss
        metrics['CE_recovered'] = ce_recovered
        metrics['alive_features'] = ((alive_features.sum() / len(alive_features)) * 100)

        # print out everything above
        print(f"Average L0 (features activated): {avg_l0:.6f}")
        print(f"Average L0 (features activated) per CLS token: {avg_l0_cls:.6f}")
        print(f"Average L0 (features activated) per image: {avg_l0_image:.6f}")
        print(f"Average Cosine Similarity: {avg_cos_sim:.4f}")
        print(f"Average Loss: {avg_loss:.6f}")
        print(f"Average Reconstruction Loss: {avg_reconstruction_loss:.6f}")
        print(f"Average Zero Ablation Loss: {avg_zero_abl_loss:.6f}")
        print(f"Average CE Score: {avg_score:.6f}")
        print(f"% CE recovered: {ce_recovered:.6f}")
        print(f"% Alive features: {(alive_features.sum() / len(alive_features)) * 100:.6f}")
        print(f"Average Reconstruction Cosine Similarity: {total_recons_cosine_similarity:.6f}")

        ind_to_name = get_imagenet_index_to_name()

        for feature_idx in random_feature_indices:

            top_mean_list = top_mean_activations_per_feature[feature_idx]

            folder = os.path.join(f'{path}/mean_feature_images', f"feature_{feature_idx}")

            os.makedirs(folder, exist_ok=True)
            
            for ix, sampling_type in enumerate([top_mean_list]):

                images = []
                labels = []
                activation_values = []
                image_indices = []

                for activation_value, image_idx in sampling_type:
                    image, label_i, idx_in_dataset, label = self.validation_dataset[image_idx]
                    images.append(image)
                    labels.append(label_i)
                    activation_values.append(activation_value)
                    image_indices.append(image_idx)

                self.save_image_grid(images, labels, activation_values, image_indices, ind_to_name, folder, feature_idx)

        return metrics