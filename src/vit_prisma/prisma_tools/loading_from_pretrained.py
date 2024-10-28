"""
Prisma Repo
By Sonia Joseph

Copyright (c) Sonia Joseph. All rights reserved.

Inspired by TransformerLens. Some functions have been adapted from the TransformerLens project.
For more information on TransformerLens, visit: https://github.com/neelnanda-io/TransformerLens
"""

import logging

from transformers import AutoConfig, ViTForImageClassification, VivitForVideoClassification, CLIPModel, ViTModel

import timm
from vit_prisma.configs.HookedViTConfig import HookedViTConfig

import torch

from typing import Dict

import einops

from typing import Union

# import partial
from functools import partial

try:
    from huggingface_hub import hf_hub_download
    hf_hub_download = partial(hf_hub_download, library_name="open_clip", library_version='2.20.0')
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False

import json

def convert_open_clip_weights(
        old_state_dict,
        cfg: HookedViTConfig,
        device = 'cuda',
):
    new_vision_model_state_dict = {}

    # Convert embedding layers
    new_vision_model_state_dict["cls_token"] = old_state_dict["visual.class_embedding"].unsqueeze(0).unsqueeze(0)
    new_vision_model_state_dict["pos_embed.W_pos"] = old_state_dict["visual.positional_embedding"].clone()


    new_vision_model_state_dict["embed.proj.weight"] = old_state_dict["visual.conv1.weight"] # Flatten convolutional embedding
    new_vision_model_state_dict["embed.proj.bias"] = torch.zeros((cfg.d_model,))

    # Convert layer norms
    new_vision_model_state_dict["ln_final.w"] = old_state_dict["visual.ln_post.weight"]
    new_vision_model_state_dict["ln_final.b"] = old_state_dict["visual.ln_post.bias"]

    new_vision_model_state_dict["ln_pre.w"] = old_state_dict["visual.ln_pre.weight"]
    new_vision_model_state_dict["ln_pre.b"] = old_state_dict["visual.ln_pre.bias"]

    print("visual projection shape", old_state_dict["visual.proj"].shape)


    #  print layernorm weights finral

    # Convert transformer blocks
    for layer in range(cfg.n_layers):
        old_layer_key = f"visual.transformer.resblocks.{layer}"
        new_layer_key = f"blocks.{layer}"

        # Layer norms
        new_vision_model_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[f"{old_layer_key}.ln_1.weight"]
        new_vision_model_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[f"{old_layer_key}.ln_1.bias"]
        new_vision_model_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[f"{old_layer_key}.ln_2.weight"]
        new_vision_model_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[f"{old_layer_key}.ln_2.bias"]

        # Attention weights
        in_proj_weight = old_state_dict[f"{old_layer_key}.attn.in_proj_weight"]
        in_proj_bias = old_state_dict[f"{old_layer_key}.attn.in_proj_bias"]
        
        # Split in_proj_weight and in_proj_bias into Q, K, V
        W_Q, W_K, W_V = in_proj_weight.chunk(3)
        b_Q, b_K, b_V = in_proj_bias.chunk(3)

        # Reshape Q, K, V weights
        W_Q = einops.rearrange(W_Q, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_K = einops.rearrange(W_K, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_V = einops.rearrange(W_V, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        
        # Reshape Q, K, V biases
        b_Q = einops.rearrange(b_Q, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_K = einops.rearrange(b_K, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_V = einops.rearrange(b_V, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)

        # Output projection
        W_O = old_state_dict[f"{old_layer_key}.attn.out_proj.weight"]
        b_O = old_state_dict[f"{old_layer_key}.attn.out_proj.bias"]
        W_O = einops.rearrange(W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)

        new_vision_model_state_dict[f"{new_layer_key}.attn.W_Q"] = W_Q
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_K"] = W_K
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_V"] = W_V
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_O"] = W_O
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_Q"] = b_Q
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_K"] = b_K
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_V"] = b_V
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        # MLP weights
        mlp_W_in = old_state_dict[f"{old_layer_key}.mlp.c_fc.weight"]
        mlp_W_out = old_state_dict[f"{old_layer_key}.mlp.c_proj.weight"]
        mlp_b_in = old_state_dict[f"{old_layer_key}.mlp.c_fc.bias"]
        mlp_b_out = old_state_dict[f"{old_layer_key}.mlp.c_proj.bias"]

        mlp_W_in = einops.rearrange(mlp_W_in, "m d -> d m")
        mlp_W_out = einops.rearrange(mlp_W_out, "d m -> m d")

        new_vision_model_state_dict[f"{new_layer_key}.mlp.W_in"] = mlp_W_in
        new_vision_model_state_dict[f"{new_layer_key}.mlp.W_out"] = mlp_W_out
        new_vision_model_state_dict[f"{new_layer_key}.mlp.b_in"] = mlp_b_in
        new_vision_model_state_dict[f"{new_layer_key}.mlp.b_out"] = mlp_b_out

    # Set W_H to be an identity matrix
    new_vision_model_state_dict["head.W_H"] = old_state_dict['visual.proj']
    new_vision_model_state_dict["head.b_H"] = torch.zeros((cfg.n_classes,))

    # print("checking awry index again")
    # print(new_vision_model_state_dict["pos_embed.W_pos"].flatten()[8474])
    # print(old_state_dict["visual.positional_embedding"].flatten()[8474])

    return new_vision_model_state_dict

def convert_dino_weights(
        old_state_dict,
        cfg: HookedViTConfig,
):
    
    new_state_dict = {}

    new_state_dict["cls_token"] = old_state_dict["embeddings.cls_token"]
    new_state_dict["pos_embed.W_pos"] = old_state_dict["embeddings.position_embeddings"].squeeze(0)
    new_state_dict["embed.proj.weight"] = old_state_dict["embeddings.patch_embeddings.projection.weight"]
    new_state_dict["embed.proj.bias"] = old_state_dict["embeddings.patch_embeddings.projection.bias"]
    new_state_dict["ln_final.w"] = old_state_dict["layernorm.weight"]
    new_state_dict["ln_final.b"] = old_state_dict["layernorm.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"encoder.layer.{layer}"
        new_layer_key = f"blocks.{layer}"
        new_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[f"{layer_key}.layernorm_before.weight"]
        new_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[f"{layer_key}.layernorm_before.bias"]
        new_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[f"{layer_key}.layernorm_after.weight"]
        new_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[f"{layer_key}.layernorm_after.bias"]

        W_Q = old_state_dict[f"{layer_key}.attention.attention.query.weight"]
        W_K = old_state_dict[f"{layer_key}.attention.attention.key.weight"]
        W_V = old_state_dict[f"{layer_key}.attention.attention.value.weight"]
        W_O = old_state_dict[f"{layer_key}.attention.output.dense.weight"]

        W_Q = einops.rearrange(W_Q, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_K = einops.rearrange(W_K, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_V = einops.rearrange(W_V, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_O = einops.rearrange(W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        
        new_state_dict[f"{new_layer_key}.attn.W_Q"] = W_Q
        new_state_dict[f"{new_layer_key}.attn.W_K"] = W_K
        new_state_dict[f"{new_layer_key}.attn.W_V"] = W_V
        new_state_dict[f"{new_layer_key}.attn.W_O"] = W_O

        b_Q = old_state_dict[f"{layer_key}.attention.attention.query.bias"]
        b_K = old_state_dict[f"{layer_key}.attention.attention.key.bias"]
        b_V = old_state_dict[f"{layer_key}.attention.attention.value.bias"]
        b_O = old_state_dict[f"{layer_key}.attention.output.dense.bias"]

        b_Q = einops.rearrange(b_Q, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_K = einops.rearrange(b_K, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_V = einops.rearrange(b_V, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)

        new_state_dict[f"{new_layer_key}.attn.b_Q"] = b_Q
        new_state_dict[f"{new_layer_key}.attn.b_K"] = b_K
        new_state_dict[f"{new_layer_key}.attn.b_V"] = b_V
        new_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        mlp_W_in = old_state_dict[f"{layer_key}.intermediate.dense.weight"]
        mlp_W_out = old_state_dict[f"{layer_key}.output.dense.weight"]
        mlp_b_in = old_state_dict[f"{layer_key}.intermediate.dense.bias"]
        mlp_b_out = old_state_dict[f"{layer_key}.output.dense.bias"]

        mlp_W_in = einops.rearrange(mlp_W_in, "m d -> d m")
        mlp_W_out = einops.rearrange(mlp_W_out, "d m -> m d")

        new_state_dict[f"{new_layer_key}.mlp.W_in"] = mlp_W_in
        new_state_dict[f"{new_layer_key}.mlp.W_out"] = mlp_W_out
        new_state_dict[f"{new_layer_key}.mlp.b_in"] = mlp_b_in
        new_state_dict[f"{new_layer_key}.mlp.b_out"] = mlp_b_out

    return new_state_dict


def convert_clip_weights(
        old_state_dict,
        old_head_state_dict,
        cfg: HookedViTConfig,
):
    
    new_vision_model_state_dict = {}

    new_vision_model_state_dict["cls_token"] = old_state_dict["embeddings.class_embedding"].unsqueeze(0).unsqueeze(0)
    new_vision_model_state_dict["pos_embed.W_pos"] = old_state_dict["embeddings.position_embedding.weight"]
    new_vision_model_state_dict["embed.proj.weight"] = old_state_dict["embeddings.patch_embedding.weight"]
    new_vision_model_state_dict["embed.proj.bias"] =  torch.zeros((cfg.d_model,), device=new_vision_model_state_dict["embed.proj.weight"].device)
    new_vision_model_state_dict["ln_final.w"] = old_state_dict["post_layernorm.weight"]
    new_vision_model_state_dict["ln_final.b"] = old_state_dict["post_layernorm.bias"]
    new_vision_model_state_dict["ln_pre.w"] = old_state_dict["pre_layrnorm.weight"] #typo in ClipModel
    new_vision_model_state_dict["ln_pre.b"] = old_state_dict["pre_layrnorm.bias"]


    for layer in range(cfg.n_layers):
        layer_key = f"encoder.layers.{layer}"
        new_layer_key = f"blocks.{layer}"

        new_vision_model_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[f"{layer_key}.layer_norm1.weight"]
        new_vision_model_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[f"{layer_key}.layer_norm1.bias"]
        new_vision_model_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[f"{layer_key}.layer_norm2.weight"]
        new_vision_model_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[f"{layer_key}.layer_norm2.bias"]

        W_Q = old_state_dict[f"{layer_key}.self_attn.q_proj.weight"]
        W_K = old_state_dict[f"{layer_key}.self_attn.k_proj.weight"]
        W_V = old_state_dict[f"{layer_key}.self_attn.v_proj.weight"]
        W_O = old_state_dict[f"{layer_key}.self_attn.out_proj.weight"]

        W_Q = einops.rearrange(W_Q, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_K = einops.rearrange(W_K, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_V = einops.rearrange(W_V, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_O = einops.rearrange(W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_Q"] = W_Q
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_K"] = W_K
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_V"] = W_V
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_O"] = W_O

        b_Q = old_state_dict[f"{layer_key}.self_attn.q_proj.bias"]
        b_K = old_state_dict[f"{layer_key}.self_attn.k_proj.bias"]
        b_V = old_state_dict[f"{layer_key}.self_attn.v_proj.bias"]
        b_O = old_state_dict[f"{layer_key}.self_attn.out_proj.bias"]

        b_Q = einops.rearrange(b_Q, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_K = einops.rearrange(b_K, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_V = einops.rearrange(b_V, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)

        new_vision_model_state_dict[f"{new_layer_key}.attn.b_Q"] = b_Q
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_K"] = b_K
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_V"] = b_V
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        mlp_W_in = old_state_dict[f"{layer_key}.mlp.fc1.weight"]
        mlp_W_out = old_state_dict[f"{layer_key}.mlp.fc2.weight"]
        mlp_b_in = old_state_dict[f"{layer_key}.mlp.fc1.bias"]
        mlp_b_out = old_state_dict[f"{layer_key}.mlp.fc2.bias"]

        mlp_W_in = einops.rearrange(mlp_W_in, "m d -> d m")
        mlp_W_out = einops.rearrange(mlp_W_out, "d m -> m d")

        new_vision_model_state_dict[f"{new_layer_key}.mlp.W_in"] = mlp_W_in
        new_vision_model_state_dict[f"{new_layer_key}.mlp.W_out"] = mlp_W_out
        new_vision_model_state_dict[f"{new_layer_key}.mlp.b_in"] = mlp_b_in
        new_vision_model_state_dict[f"{new_layer_key}.mlp.b_out"] = mlp_b_out


    new_vision_model_state_dict["head.W_H"] = einops.rearrange(old_head_state_dict["weight"], "c d -> d c")
    new_vision_model_state_dict["head.b_H"] = torch.zeros((cfg.n_classes,), device=new_vision_model_state_dict["head.W_H"].device)

        
    return new_vision_model_state_dict


def convert_timm_weights(
        old_state_dict,
        cfg: HookedViTConfig,
):

    new_state_dict = {}
    new_state_dict["cls_token"] = old_state_dict["cls_token"]
    new_state_dict["pos_embed.W_pos"] = old_state_dict["pos_embed"].squeeze(0)
    new_state_dict["embed.proj.weight"] = old_state_dict["patch_embed.proj.weight"]
    new_state_dict["embed.proj.bias"] = old_state_dict["patch_embed.proj.bias"] 
    new_state_dict["ln_final.w"] = old_state_dict["norm.weight"]
    new_state_dict["ln_final.b"] = old_state_dict["norm.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"blocks.{layer}" 
        new_state_dict[f"{layer_key}.ln1.w"] = old_state_dict[f"{layer_key}.norm1.weight"]
        new_state_dict[f"{layer_key}.ln1.b"] = old_state_dict[f"{layer_key}.norm1.bias"]
        new_state_dict[f"{layer_key}.ln2.w"] = old_state_dict[f"{layer_key}.norm2.weight"]
        new_state_dict[f"{layer_key}.ln2.b"] = old_state_dict[f"{layer_key}.norm2.bias"]

        W = old_state_dict[f"{layer_key}.attn.qkv.weight"]
        W_reshape = einops.rearrange( W, "(three h dh) d ->three h d dh" , three=3, h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_Q, W_K, W_V = torch.unbind(W_reshape, dim=0)
        new_state_dict[f"{layer_key}.attn.W_Q"] = W_Q
        new_state_dict[f"{layer_key}.attn.W_K"] = W_K
        new_state_dict[f"{layer_key}.attn.W_V"] = W_V

        W_O = old_state_dict[f"{layer_key}.attn.proj.weight"]
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        new_state_dict[f"{layer_key}.attn.W_O"] = W_O

        attn_bias = old_state_dict[f"{layer_key}.attn.qkv.bias"]
        attn_bias_reshape = einops.rearrange(attn_bias, "(three h dh) -> three h dh", three=3, h=cfg.n_heads, dh=cfg.d_head)
        b_Q, b_K, b_V = torch.unbind(attn_bias_reshape, dim=0)
        new_state_dict[f"{layer_key}.attn.b_Q"] = b_Q
        new_state_dict[f"{layer_key}.attn.b_K"] = b_K
        new_state_dict[f"{layer_key}.attn.b_V"] = b_V

        b_O = old_state_dict[f"{layer_key}.attn.proj.bias"]
        new_state_dict[f"{layer_key}.attn.b_O"] = b_O

        new_state_dict[f"{layer_key}.mlp.b_in"] = old_state_dict[f"{layer_key}.mlp.fc1.bias"]
        new_state_dict[f"{layer_key}.mlp.b_out"] = old_state_dict[f"{layer_key}.mlp.fc2.bias"]

        mlp_W_in = old_state_dict[f"{layer_key}.mlp.fc1.weight"]
        mlp_W_in = einops.rearrange(mlp_W_in, "m d -> d m")
        new_state_dict[f"{layer_key}.mlp.W_in"] = mlp_W_in

        mlp_W_out = old_state_dict[f"{layer_key}.mlp.fc2.weight"]
        mlp_W_out = einops.rearrange(mlp_W_out, "d m -> m d")
        new_state_dict[f"{layer_key}.mlp.W_out"] = mlp_W_out


    new_state_dict["head.W_H"] = einops.rearrange(old_state_dict["head.weight"], "c d -> d c")
    new_state_dict["head.b_H"] = old_state_dict["head.bias"]


    return new_state_dict


def convert_vivet_weights(
        old_state_dict,
        cfg: HookedViTConfig,
):

    new_state_dict = {}

    new_state_dict["cls_token"] = old_state_dict["vivit.embeddings.cls_token"]
    new_state_dict["pos_embed.W_pos"] = old_state_dict["vivit.embeddings.position_embeddings"].squeeze(0)
    new_state_dict["embed.proj.weight"] = old_state_dict["vivit.embeddings.patch_embeddings.projection.weight"]
    new_state_dict["embed.proj.bias"] = old_state_dict["vivit.embeddings.patch_embeddings.projection.bias"] 
    new_state_dict["ln_final.w"] = old_state_dict["vivit.layernorm.weight"]
    new_state_dict["ln_final.b"] = old_state_dict["vivit.layernorm.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"vivit.encoder.layer.{layer}" 
        new_layer_key = f"blocks.{layer}"
        new_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[f"{layer_key}.layernorm_before.weight"]
        new_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[f"{layer_key}.layernorm_before.bias"]
        new_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[f"{layer_key}.layernorm_after.weight"]
        new_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[f"{layer_key}.layernorm_after.bias"]

        W_Q  = old_state_dict[f"{layer_key}.attention.attention.query.weight"]
        W_K  = old_state_dict[f"{layer_key}.attention.attention.key.weight"]
        W_V  = old_state_dict[f"{layer_key}.attention.attention.value.weight"]

        new_state_dict[f"{new_layer_key}.attn.W_Q"] = einops.rearrange(W_Q, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        new_state_dict[f"{new_layer_key}.attn.W_K"] = einops.rearrange(W_K, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        new_state_dict[f"{new_layer_key}.attn.W_V"] = einops.rearrange(W_V, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)

        W_O = old_state_dict[f"{layer_key}.attention.output.dense.weight"]
        new_state_dict[f"{new_layer_key}.attn.W_O"] =  einops.rearrange(W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)


        b_Q  = old_state_dict[f"{layer_key}.attention.attention.query.bias"]
        b_K  = old_state_dict[f"{layer_key}.attention.attention.key.bias"]
        b_V  = old_state_dict[f"{layer_key}.attention.attention.value.bias"]

        new_state_dict[f"{new_layer_key}.attn.b_Q"] = einops.rearrange(b_Q, "(h dh) -> h dh",  h=cfg.n_heads, dh=cfg.d_head)
        new_state_dict[f"{new_layer_key}.attn.b_K"] = einops.rearrange(b_K, "(h dh) -> h dh",  h=cfg.n_heads, dh=cfg.d_head)
        new_state_dict[f"{new_layer_key}.attn.b_V"] = einops.rearrange(b_V, "(h dh) -> h dh",  h=cfg.n_heads, dh=cfg.d_head)

        b_O = old_state_dict[f"{layer_key}.attention.output.dense.bias"]
        new_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        mlp_W_in = old_state_dict[f"{layer_key}.intermediate.dense.weight"]
        new_state_dict[f"{new_layer_key}.mlp.W_in"] =  einops.rearrange(mlp_W_in, "m d -> d m")

        mlp_W_out  = old_state_dict[f"{layer_key}.output.dense.weight"]
       
        new_state_dict[f"{new_layer_key}.mlp.W_out"] = einops.rearrange(mlp_W_out, "d m -> m d")

        new_state_dict[f"{new_layer_key}.mlp.b_in"] =  old_state_dict[f"{layer_key}.intermediate.dense.bias"]
        new_state_dict[f"{new_layer_key}.mlp.b_out"] =  old_state_dict[f"{layer_key}.output.dense.bias"]


    new_state_dict["head.W_H"] = einops.rearrange(old_state_dict["classifier.weight"], "c d -> d c")
    new_state_dict["head.b_H"] = old_state_dict["classifier.bias"]



    return new_state_dict

def convert_hf_vit_for_image_classification_weights(   old_state_dict,
        cfg: HookedViTConfig,
):

    new_state_dict = {}

    #exit(0)
    new_state_dict["cls_token"] = old_state_dict["vit.embeddings.cls_token"]
    new_state_dict["pos_embed.W_pos"] = old_state_dict["vit.embeddings.position_embeddings"].squeeze(0)
    new_state_dict["embed.proj.weight"] = old_state_dict["vit.embeddings.patch_embeddings.projection.weight"]
    new_state_dict["embed.proj.bias"] = old_state_dict["vit.embeddings.patch_embeddings.projection.bias"] 
    new_state_dict["ln_final.w"] = old_state_dict["vit.layernorm.weight"]
    new_state_dict["ln_final.b"] = old_state_dict["vit.layernorm.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"vit.encoder.layer.{layer}" 
        new_layer_key = f"blocks.{layer}"
        new_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[f"{layer_key}.layernorm_before.weight"]
        new_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[f"{layer_key}.layernorm_before.bias"]
        new_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[f"{layer_key}.layernorm_after.weight"]
        new_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[f"{layer_key}.layernorm_after.bias"]

        W_Q  = old_state_dict[f"{layer_key}.attention.attention.query.weight"]
        W_K  = old_state_dict[f"{layer_key}.attention.attention.key.weight"]
        W_V  = old_state_dict[f"{layer_key}.attention.attention.value.weight"]

        new_state_dict[f"{new_layer_key}.attn.W_Q"] = einops.rearrange(W_Q, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        new_state_dict[f"{new_layer_key}.attn.W_K"] = einops.rearrange(W_K, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        new_state_dict[f"{new_layer_key}.attn.W_V"] = einops.rearrange(W_V, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)

        W_O = old_state_dict[f"{layer_key}.attention.output.dense.weight"]
        new_state_dict[f"{new_layer_key}.attn.W_O"] =  einops.rearrange(W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)


        b_Q  = old_state_dict[f"{layer_key}.attention.attention.query.bias"]
        b_K  = old_state_dict[f"{layer_key}.attention.attention.key.bias"]
        b_V  = old_state_dict[f"{layer_key}.attention.attention.value.bias"]

        new_state_dict[f"{new_layer_key}.attn.b_Q"] = einops.rearrange(b_Q, "(h dh) -> h dh",  h=cfg.n_heads, dh=cfg.d_head)
        new_state_dict[f"{new_layer_key}.attn.b_K"] = einops.rearrange(b_K, "(h dh) -> h dh",  h=cfg.n_heads, dh=cfg.d_head)
        new_state_dict[f"{new_layer_key}.attn.b_V"] = einops.rearrange(b_V, "(h dh) -> h dh",  h=cfg.n_heads, dh=cfg.d_head)

        b_O = old_state_dict[f"{layer_key}.attention.output.dense.bias"]
        new_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        mlp_W_in = old_state_dict[f"{layer_key}.intermediate.dense.weight"]
        new_state_dict[f"{new_layer_key}.mlp.W_in"] =  einops.rearrange(mlp_W_in, "m d -> d m")

        mlp_W_out  = old_state_dict[f"{layer_key}.output.dense.weight"]
       
        new_state_dict[f"{new_layer_key}.mlp.W_out"] = einops.rearrange(mlp_W_out, "d m -> m d")

        new_state_dict[f"{new_layer_key}.mlp.b_in"] =  old_state_dict[f"{layer_key}.intermediate.dense.bias"]
        new_state_dict[f"{new_layer_key}.mlp.b_out"] =  old_state_dict[f"{layer_key}.output.dense.bias"]

    new_state_dict["head.W_H"] = einops.rearrange(old_state_dict["classifier.weight"], "c d -> d c")
    new_state_dict["head.b_H"] = old_state_dict["classifier.bias"]



    return new_state_dict


def convert_open_clip_config(model_cfg):
    cfg = HookedViTConfig()
    cfg.d_model = model_cfg['vision_cfg']['width']
    cfg.n_layers = model_cfg['vision_cfg']['layers']
    cfg.patch_size = model_cfg['vision_cfg']['patch_size']
    cfg.image_size = model_cfg['vision_cfg']['image_size']
    cfg.d_mlp = cfg.d_model * 4
    cfg.n_heads = 12
    cfg.d_head = cfg.d_model // cfg.n_heads
    cfg.n_classes = model_cfg['embed_dim'] # This is the projection dimensionality
    cfg.return_type = None
    cfg.layer_norm_pre = True
    cfg.eps = 1e-5
    cfg.normalization_type = "LN"
    cfg.use_cls_token = True
    cfg.normalize_output = True
    return cfg


def get_pretrained_state_dict(
    official_model_name: str,
    is_timm: bool,
    is_clip: bool,
    cfg: HookedViTConfig,
    hf_model=None,
    dtype: torch.dtype = torch.float32,
    return_old_state_dict=False,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Loads in the model weights for a pretrained model, and processes them to
    have the HookedTransformer parameter names and shapes. Supports checkpointed
    models (and expects the checkpoint info to be stored in the config object)

    hf_model: Optionally, a HuggingFace model object. If provided, we will use
        these weights rather than reloading the model.
    dtype: The dtype to load the HuggingFace model in.
    kwargs: Other optional arguments passed to HuggingFace's from_pretrained.
        Also given to other HuggingFace functions when compatible.
    """
    if "torch_dtype" in kwargs:
        dtype = kwargs["torch_dtype"]
        del kwargs["torch_dtype"]
    # official_model_name = get_official_model_name(official_model_name)
    # if official_model_name.startswith(NEED_REMOTE_CODE_MODELS) and not kwargs.get(
    #     "trust_remote_code", False
    # ):
    #     logging.warning(
    #         f"Loading model {official_model_name} state dict requires setting trust_remote_code=True"
    #     )
    #     kwargs["trust_remote_code"] = True
    if 'dino' in official_model_name:
        is_timm = False
        
    try:
        print("Official model name", official_model_name)
        if is_timm:
            hf_model = hf_model if hf_model is not None else timm.create_model(official_model_name, pretrained=True)
            for param in hf_model.parameters():
                param.requires_grad = False
            state_dict = convert_timm_weights(hf_model.state_dict(), cfg)
        elif is_clip and official_model_name.startswith("open-clip:"):
            print("Converting OpenCLIP weights")
            checkpoint_path = download_pretrained_from_hf(remove_open_clip_prefix(official_model_name), filename='open_clip_pytorch_model.bin')
            old_state_dict = load_state_dict(checkpoint_path)
            state_dict = convert_open_clip_weights(old_state_dict, cfg)
        elif is_clip:
            full_model = hf_model if hf_model is not None else CLIPModel.from_pretrained(official_model_name)
            for param in full_model.parameters():
                param.requires_grad = False
            vision = full_model.vision_model
            visual_projection = full_model.visual_projection
            state_dict = convert_clip_weights(vision.state_dict(), visual_projection.state_dict(), cfg)
        elif cfg.is_video_transformer:
            if "vivit" in official_model_name:
                hf_model = hf_model if hf_model is not None else VivitForVideoClassification.from_pretrained(official_model_name, torch_dtype=dtype, **kwargs)

                for param in hf_model.parameters():
                    param.requires_grad = False

                state_dict = convert_vivet_weights(hf_model.state_dict(), cfg)
            else:
                raise ValueError

        elif 'dino' in official_model_name:
            hf_model = hf_model if hf_model is not None else ViTModel.from_pretrained(official_model_name, torch_dtype=dtype, **kwargs)
            for param in hf_model.parameters():
                param.requires_grad = False
            state_dict = convert_dino_weights(hf_model.state_dict(), cfg)
        
        else:
            hf_model = hf_model if hf_model is not None else ViTForImageClassification.from_pretrained(
                    official_model_name, torch_dtype=dtype, **kwargs
            )
            state_dict = convert_hf_vit_for_image_classification_weights(hf_model.state_dict(), cfg)

            # Load model weights, and fold in layer norm weights
                

        return state_dict

    except:
        raise ValueError(
            f"Loading weights from the architecture is not currently supported: {cfg.original_architecture}, generated from model name {cfg.model_name}. Feel free to open an issue on GitHub to request this feature."
        )

def fill_missing_keys(model, state_dict):
    """Takes in a state dict from a pretrained model, and fills in any missing keys with the default initialization.

    This function is assumed to be run before weights are initialized.

    Args:
        state_dict (dict): State dict from a pretrained model

    Returns:
        dict: State dict with missing keys filled in
    """
    # Get the default state dict
    default_state_dict = model.state_dict()
    # Get the keys that are missing from the pretrained model
    missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())
    # Fill in the missing keys with the default initialization
    for key in missing_keys:
        if "hf_model" in key:
            # Skip keys that are from the HuggingFace model, if loading from HF.
            continue
        if "W_" in key:
            logging.warning(
                "Missing key for a weight matrix in pretrained, filled in with an empty tensor: {}".format(
                    key
                )
            )
        state_dict[key] = default_state_dict[key]
    return state_dict


def remove_open_clip_prefix(text, prefix="open-clip:"):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text 

def convert_pretrained_model_config(model_name: str, is_timm: bool = True, is_clip: bool = False) -> HookedViTConfig:
    if 'dino' in model_name:
        is_timm = False
        
    if is_timm:
        model = timm.create_model(model_name)
        hf_config = AutoConfig.from_pretrained(model.default_cfg['hf_hub_id'])
    elif is_clip and model_name.startswith("open-clip"): # OpenCLIP models
        config_path = download_pretrained_from_hf(remove_open_clip_prefix(model_name), filename='open_clip_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            pretrained_cfg = config['preprocess_cfg']
            hf_config = config['model_cfg']
        hf_config = convert_open_clip_config(hf_config)
        return hf_config
    elif is_clip: # Extract vision encoder from dual-encoder CLIP model. HF models
        hf_config = AutoConfig.from_pretrained(model_name).vision_config
        hf_config.architecture = 'vit_clip_vision_encoder'
        hf_config.num_classes = hf_config.projection_dim # final output dimension instead of classes
    else:
        hf_config = AutoConfig.from_pretrained(model_name)

    if hasattr(hf_config, 'patch_size'):
        ps = hf_config.patch_size
    elif hasattr(hf_config, "tubelet_size"):
        ps = hf_config.tubelet_size[1]

    pretrained_config = {
                    'n_layers' : hf_config.num_hidden_layers,
                    'd_model' : hf_config.hidden_size,
                    'd_head' : hf_config.hidden_size // hf_config.num_attention_heads,
                    'model_name' : hf_config._name_or_path,
                    'n_heads' : hf_config.num_attention_heads,
                    'd_mlp' : hf_config.intermediate_size,
                    'activation_name' : hf_config.hidden_act,
                    'eps' : hf_config.layer_norm_eps,
                    'original_architecture' : getattr(hf_config, 'architecture', getattr(hf_config, 'architectures', None)),
                    'initializer_range' : hf_config.initializer_range,
                    'n_channels' : hf_config.num_channels,
                    'patch_size' : ps,
                    'image_size' : hf_config.image_size,
                    'n_classes' : getattr(hf_config, "num_classes", None),
                    'n_params' : sum(p.numel() for p in model.parameters() if p.requires_grad) if is_timm else None,
                }
    
    # Rectifying Huggingface bugs:
    # Currently a bug getting configs, only this model confirmed to work and even it requires modification of eps
    if is_timm and model_name == "vit_base_patch16_224":
        pretrained_config.update({
            "eps": 1e-6,
            "return_type": "class_logits",
        })
    
    # Config for 32 is incorrect, fix manually 
    if is_timm and model_name == "vit_base_patch32_224":
        pretrained_config.update({
            "patch_size": 32,
            "eps": 1e-6,
            "return_type": "class_logits"
        })

    if is_clip:
        pretrained_config.update({
            "layer_norm_pre": True,
            "return_type": "class_logits" # actually returns 'visual_projection'
        })

    if "dino" in model_name:
        pretrained_config.update({
            "return_type": "pre_logits",
            "n_classes": 768,
        })

    # Config is for ViVet, need to add more properties
    if hasattr(hf_config, "tubelet_size"):
        pretrained_config.update({
            "is_video_transformer": True,
            "video_tubelet_depth": hf_config.tubelet_size[0],
            "video_num_frames": hf_config.video_size[0],
            "n_classes": 400 if "kinetics400" in model_name else None,
            "return_type": "class_logits" if "kinetics400" in model_name else "pre_logits",

        })

    if pretrained_config['n_classes'] is None:
        id2label = getattr(hf_config, "id2label", None)
        if id2label is not None:
            pretrained_config.update({
                "n_classes": len(id2label),
                "return_type": "class_logits"
            })
    
    return HookedViTConfig.from_dict(pretrained_config)


def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.')
    return _has_hf_hub


def download_pretrained_from_hf(
        model_id: str,
        filename: str = 'open_clip_pytorch_model.bin',
        revision=None,
        cache_dir: Union[str, None] = None,
):
    print("model_id download_pretrained_from_hf:", model_id)
    has_hf_hub(True)
    cached_file = hf_hub_download(model_id, filename, revision=revision, cache_dir=cache_dir)
    return cached_file

def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

# checkpoint_path = download_pretrained_from_hf('laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K', filename='open_clip_pytorch_model.bin')
# config_path = download_pretrained_from_hf('laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K', filename='open_clip_config.json')

# with open(config_path, 'r', encoding='utf-8') as f:
#     config = json.load(f)
#     pretrained_cfg = config['preprocess_cfg']
#     model_cfg = config['model_cfg']

#     print(pretrained_cfg)
#     print(model_cfg)

# state_dict = load_state_dict(checkpoint_path)

# print("old state dictionary")
# for key in state_dict:
#     print(key, state_dict[key].shape)

# new_cfg = convert_open_clip_config(model_cfg)
# new_state_dict = convert_open_clip_weights(state_dict, new_cfg)

# print()
# print("new state dictionary")
# for key in new_state_dict:
#     print(key, new_state_dict[key].shape)


