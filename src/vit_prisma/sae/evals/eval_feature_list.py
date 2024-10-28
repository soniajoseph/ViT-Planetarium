import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from vit_prisma.sae.evals.evals import highest_activating_tokens, get_heatmap, image_patch_heatmap, setup_environment, load_model, load_dataset, EvalConfig
from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_index_to_name

from vit_prisma.sae.sae import SparseAutoencoder

import argparse


def sample_features(feature_ids, cfg, model, sparse_autoencoder, val_dataloader, val_data, val_data_visualize):
    this_max = cfg.eval_max
    # feature_ids = [int(i[0]) for i in feature_ids]
    print(f"Sampling for features {feature_ids}...")

    max_indices = {i: None for i in feature_ids}
    max_values = {i: None for i in feature_ids}
    b_enc = sparse_autoencoder.b_enc[feature_ids]
    W_enc = sparse_autoencoder.W_enc[:, feature_ids]
    
    for batch_idx, (total_images, total_labels, total_indices) in tqdm(enumerate(val_dataloader), total=this_max//cfg.batch_size): 
        total_images = total_images.to(cfg.device)
        total_indices = total_indices.to(cfg.device)

        new_top_info = highest_activating_tokens(total_images, model, sparse_autoencoder, W_enc, b_enc, feature_ids)
        
        for feature_id in feature_ids:
            feature_data = new_top_info[feature_id]
            batch_image_indices = torch.tensor(feature_data['image_indices'])
            token_activation_values = torch.tensor(feature_data['values'], device=cfg.device)
            global_image_indices = total_indices[batch_image_indices]

            unique_image_indices, unique_indices = torch.unique(global_image_indices, return_inverse=True)
            unique_activation_values = torch.zeros_like(unique_image_indices, dtype=torch.float, device=cfg.device)
            unique_activation_values.index_reduce_(0, unique_indices, token_activation_values, 'amax')

            if max_indices[feature_id] is None: 
                max_indices[feature_id] = unique_image_indices
                max_values[feature_id] = unique_activation_values
            else:
                all_indices = torch.cat((max_indices[feature_id], unique_image_indices))
                all_values = torch.cat((max_values[feature_id], unique_activation_values))
                
                unique_all_indices, unique_all_idx = torch.unique(all_indices, return_inverse=True)
                unique_all_values = torch.zeros_like(unique_all_indices, dtype=torch.float)
                unique_all_values.index_reduce_(0, unique_all_idx, all_values, 'amax')
                
                if len(unique_all_indices) > cfg.max_images_per_feature:
                    _, top_k_idx = torch.topk(unique_all_values, k=cfg.max_images_per_feature)
                    max_indices[feature_id] = unique_all_indices[top_k_idx]
                    max_values[feature_id] = unique_all_values[top_k_idx]
                else:
                    max_indices[feature_id] = unique_all_indices
                    max_values[feature_id] = unique_all_values

        if batch_idx*cfg.batch_size >= this_max:
            break

    top_per_feature = {i: (max_values[i].detach().cpu(), max_indices[i].detach().cpu()) for i in feature_ids}
    ind_to_name = get_imagenet_index_to_name()

    for feature_id in feature_ids:
        max_vals, max_inds = top_per_feature[feature_id]
        images = []
        model_images = []
        gt_labels = []
        unique_bids = set()
        for bid, v in zip(max_inds, max_vals):
            if len(unique_bids) >= cfg.max_images_per_feature:
                break
            if bid not in unique_bids:
                image, label, image_ind = val_data_visualize[bid]
                images.append(image)
                model_img, _, _ = val_data[bid]
                model_images.append(model_img)
                gt_labels.append(ind_to_name[str(label)][1])
                unique_bids.add(bid)
        
        grid_size = int(np.ceil(np.sqrt(len(images))))
        fig, axs = plt.subplots(int(np.ceil(len(images)/grid_size)), grid_size, figsize=(15, 15))
        name = f"Feature: {feature_id}"
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

            heatmap = get_heatmap(model_img, model, sparse_autoencoder, feature_id, cfg.device)
            heatmap = image_patch_heatmap(heatmap, cfg)
            display = image_tensor.numpy().transpose(1, 2, 0)

            axs[row, col].imshow(display)
            axs[row, col].imshow(heatmap, cmap='viridis', alpha=0.3)
            axs[row, col].set_title(f"{label} {val.item():0.06f}")
            axs[row, col].axis('off')

        plt.tight_layout()
        folder = os.path.join(cfg.max_image_output_folder, f"feature_{feature_id}")
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, f"feature_id_{feature_id}.png"))
        plt.savefig(os.path.join(folder, f"feature_id_{feature_id}.svg"))
        print(f"Saved images for feature {feature_id} in {folder}")
        plt.close()

def parse_features(features):
    return [int(f.strip()) for f in features.strip('[]').split(',')]



def load_sae(cfg):
    # sparse_autoencoder = SparseAutoencoder(cfg).load_from_pretrained(cfg.sae_path)
    sae = SparseAutoencoder(cfg).load_from_pretrained_legacy_saelens_v2(cfg.sae_path)
    return sae

# Usage example
def main(feature_id_list):
    # Initialize your cfg, model, sparse_autoencoder, val_dataloader, val_data, val_data_visualize here

    setup_environment()
    cfg = EvalConfig()
    model = load_model(cfg)
        
    cfg.sae_path = '/network/scratch/s/sonia.joseph/sae_checkpoints/tinyclip_40M_mlp_out/mustache_sae_16_mlp_out/UPDATED-final_sae_group_wkcn_TinyCLIP-ViT-40M-32-Text-19M-LAION400M_blocks.9.hook_mlp_out_8192.pt'

    sparse_autoencoder = load_sae(cfg)
    print("Loaded SAE config", sparse_autoencoder.cfg) if cfg.verbose else None
    val_data, val_data_visualize, val_dataloader = load_dataset(cfg)
    print("Loaded model and data") if cfg.verbose else None
    sample_features(feature_id_list, cfg, model, sparse_autoencoder, val_dataloader, val_data, val_data_visualize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample features from the model')
    parser.add_argument('--features', nargs='+', type=int, help='List of feature ids to sample')
    args = parser.parse_args()

    features = args.features
    # features = parse_features(args.features)
    print(f"Sampling features {features}")

    main(features)