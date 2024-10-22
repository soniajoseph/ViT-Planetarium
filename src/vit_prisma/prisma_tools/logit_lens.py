"""
Prisma Repo
By Sonia Joseph

Copyright (c) Sonia Joseph. All rights reserved.

Inspired by TransformerLens. Some functions have been adapted from the TransformerLens project.
For more information on TransformerLens, visit: https://github.com/neelnanda-io/TransformerLens
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Union, Optional, Dict, List, Tuple

from vit_prisma.utils.data_utils.imagenet.imagenet_dict import IMAGENET_DICT
from vit_prisma.utils.data_utils.imagenet.imagenet_utils import imagenet_index_from_word


def get_patch_logit_directions(cache, all_answers: torch.Tensor, incl_mid: bool = False, return_labels: bool = True) -> tuple:
    """
    Computes the patch logit directions based on accumulated residuals from the cache.

    Args:
        cache: An object that provides methods to access and process model residuals.
        all_answers (torch.Tensor): A tensor containing all possible answers with shape (num_answers, d_model).
        incl_mid (bool, optional): Whether to include intermediate layers. Default is False.
        return_labels (bool, optional): Whether to return labels along with the result. Default is True.

    Returns:
        tuple: A tuple containing:
            - result (torch.Tensor): The computed logit directions with shape (batch_size, num_patches, num_labels, num_answers).
            - labels: Labels associated with the accumulated residuals, if `return_labels` is True.
    """
    
    accumulated_residual, labels = cache.accumulated_resid(
        layer=-1, incl_mid=incl_mid, return_labels=True
    )
    scaled_residual_stack = cache.apply_ln_to_stack(
        accumulated_residual, layer=-1,
    )
    result = torch.einsum('lbpd,od -> lbpo', scaled_residual_stack, all_answers)
    # Rearrange so batches are first
    result = result.permute(1, 2, 0, 3)
    return result, labels

def get_patch_logit_dictionary(
    patch_logit_directions: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
    batch_idx: int = 0, 
    rank_label: Optional[str] = None
) -> Dict[int, List[Tuple[float, str, int, Optional[int]]]]:
    """
    Constructs a dictionary of patch logit predictions for a given batch index.

    Args:
        patch_logit_directions (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): A tensor or a tuple of tensors 
                                                                               containing the logit directions with shape 
                                                                               (batch_size, num_patches, num_labels, num_answers).
        batch_idx (int, optional): The index of the batch to process. Default is 0.
        rank_label (Optional[str], optional): A label to rank against the predictions. Default is None.

    Returns:
        Dict[int, List[Tuple[float, str, int, Optional[int]]]]: A dictionary where each key is a patch index and each value is a list of tuples.
                                                                Each tuple contains the logit, predicted class name, predicted index,
                                                                and optionally the rank of the rank_label.
    """    
    
    patch_dictionary = defaultdict(list)
    # if tuple, get first entry
    if isinstance(patch_logit_directions, tuple):
        patch_logit_directions = patch_logit_directions[0]
    # Go through laeyrs of one batch
    for patch_idx, patches in enumerate(patch_logit_directions[batch_idx]):
        # Go through every patch and get max prediction
        for logits in patches:
            probs = torch.softmax(logits, dim=-1)
            # Get index of max prediction
            predicted_idx = int(torch.argmax(probs))
            logit = logits[predicted_idx].item()
            predicted_class_name = IMAGENET_DICT[predicted_idx]
            if rank_label:
                # Where is the rank_label in the sorted list?
                rank_index = imagenet_index_from_word(rank_label)
                sorted_list = torch.argsort(probs, descending=True)
                rank = np.where(sorted_list == rank_index)[0][0]
                patch_dictionary[patch_idx].append((logit, predicted_class_name, predicted_idx, rank))
            else:
                patch_dictionary[patch_idx].append((logit, predicted_class_name, predicted_idx))
    return patch_dictionary
