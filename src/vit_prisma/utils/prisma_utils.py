"""
Prisma Repo
By Sonia Joseph

Copyright (c) Sonia Joseph. All rights reserved.

Inspired by TransformerLens. Some functions have been adapted from the TransformerLens project.
For more information on TransformerLens, visit: https://github.com/neelnanda-io/TransformerLens
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import torch
from jaxtyping import Float, Int
import numpy as np
import re
import json
import logging

from vit_prisma.utils.data_utils.imagenet.imagenet_dict import IMAGENET_DICT
from vit_prisma.utils.data_utils.imagenet.imagenet_utils import imagenet_index_from_word


def test_prompt(example_data_point: torch.Tensor, model: Any, example_answer: Optional[str] = None, top_k: int = 10) -> None:
    """
    Evaluates a model's predictions on a given data point and prints the top-k predicted labels along with their logits and probabilities.

    Args:
        example_data_point (torch.Tensor): The input data point to be evaluated by the model.
        model (Any): The model used for generating predictions.
        example_answer (Optional[str], optional): The correct label for the data point, if available. Default is None.
        top_k (int, optional): The number of top predictions to display. Default is 10.

    Returns:
        None
    """

    logits = model(example_data_point.unsqueeze(0))
    probs = logits.softmax(dim=-1)
    probs = probs.squeeze(0).detach().numpy()
    sorted_probs = np.sort(probs)[::-1]
    sorted_probs_args = np.argsort(probs)[::-1]

    for i in range(top_k):
        index = sorted_probs_args[i]
        prob = sorted_probs[i]
        logit = logits[0, index].item()  # Assuming you want to show the original logit value
        label = IMAGENET_DICT[index]  # Adjust based on your mapping

        rank_str = f"Top {i}th token."
        logit_str = f"Logit: {logit:.2f}"
        prob_str = f"Prob: {prob * 100:.2f}%"
        token_str = f"Label: |{label}|"

        print(f"{rank_str} {logit_str} {prob_str} {token_str}")

    if example_answer:
      tabby_cat_idx = imagenet_index_from_word(example_answer)

    # Example for displaying ranks of the answer tokens, adjust according to your needs
      answer_index = imagenet_index_from_word(example_answer)
      answer_indices = [answer_index]  # Assuming index is the answer index, adjust as necessary
      print("Rank of the correct answer:")
      for ans_index in answer_indices:
          rank = np.where(sorted_probs_args == ans_index)[0][0]
          print(f"Class Name: {example_answer} | Rank: {rank} | ImageNet Index: {tabby_cat_idx}")

def transpose(tensor: Float[torch.Tensor, "... a b"]) -> Float[torch.Tensor, "... b a"]:
    """
    Utility to swap the last two dimensions of a tensor, regardless of the number of leading dimensions
    """
    return tensor.transpose(-1, -2)

# Type alias
SliceInput: Type = Optional[
    Union[
        int,
        Tuple[int,],
        Tuple[int, int],
        Tuple[int, int, int],
        List[int],
        torch.Tensor,
        np.ndarray,
    ]
]
"""An object that represents a slice input. It can be a tuple of integers or a slice object.

An optional type alias for a slice input used in the `ActivationCache` module.

A `SliceInput` can be one of the following types:
    - `int`: an integer representing a single position
    - `Tuple[int, int]`: a tuple of two integers representing a range of positions
    - `Tuple[int, int, int]`: a tuple of three integers representing a range of positions with a step size
    - `List[int]`: a list of integers representing multiple positions
    - `torch.Tensor`: a tensor containing a boolean mask or a list of indices to be selected from the input tensor.

`SliceInput` is used in the `apply_ln_to_stack` method in the `ActivationCache` module.
"""

class Slice:
    """An object that represents a slice input. It can be a tuple of integers or a slice object.

    We use a custom slice syntax because Python/Torch's don't let us reduce the number of dimensions:

    Note that slicing with input_slice=None means do nothing, NOT add an extra dimension (use unsqueeze for that)

    There are several modes:
    int - just index with that integer (decreases number of dimensions)
    slice - Input is a tuple converted to a slice ((k,) means :k, (k, m) means m:k, (k, m, n) means m:k:n)
    array - Input is a list or tensor or numpy array, converted to a numpy array, and we take the stack of values at those indices
    identity - Input is None, leave it unchanged.

    Examples for dim=0:
    if input_slice=0, tensor -> tensor[0]
    elif input_slice = (1, 5), tensor -> tensor[1:5]
    elif input_slice = (1, 5, 2), tensor -> tensor[1:5:2] (ie indexing with [1, 3])
    elif input_slice = [1, 4, 5], tensor -> tensor[[1, 4, 5]] (ie changing the first axis to have length 3, and taking the indices 1, 4, 5 out).
    elif input_slice is a Tensor, same as list - Tensor is assumed to be a 1D list of indices.
    """

    def __init__(
        self,
        input_slice: SliceInput = None,
    ):
        """
        Modular component for slicing tensors. Can be used to slice a tensor along a given dimension, or to index into a tensor along a given dimension.

        Args:
            input_slice (SliceInput): The slice to apply. Can be an int, a tuple, a list, a torch.Tensor, or None. If None, do nothing.

        Raises:
            ValueError: If the input_slice is not one of the above types.
        """
        if type(input_slice) == tuple:
            input_slice: slice = slice(*input_slice)
            self.slice = input_slice
            self.mode = "slice"
        elif type(input_slice) == int:
            self.slice = input_slice
            self.mode = "int"
        elif type(input_slice) == slice:
            self.slice = input_slice
            self.mode = "slice"
        elif type(input_slice) in [list, torch.Tensor, np.ndarray]:
            self.slice = to_numpy(input_slice)
            self.mode = "array"
        elif input_slice is None:
            self.slice = slice(None)
            self.mode = "identity"
        else:
            raise ValueError(f"Invalid input_slice {input_slice}")
        

    def apply(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """
        Takes in a tensor and a slice, and applies the slice to the given dimension (supports positive and negative dimension syntax). Returns the sliced tensor.

        Args:
            tensor (torch.Tensor): The tensor to slice.
            dim (int, optional): The dimension to slice along. Supports positive and negative dimension syntax.

        Returns:
            torch.Tensor: The sliced tensor.
        """
        ndim = tensor.ndim
        slices = [slice(None)] * ndim
        slices[dim] = self.slice
        return tensor[tuple(slices)]

    def indices(
        self,
        max_ctx: Optional[int] = None,
    ) -> Union[np.ndarray, np.int32, np.int64]:
        """
        Returns the indices when this slice is applied to an axis of size max_ctx. Returns them as a numpy array, for integer slicing it is eg array([4])

        Args:
            max_ctx (int, optional): The size of the axis to slice. Only used if the slice is not an integer.

        Returns:
            np.ndarray: The indices that this slice will select.

        Raises:
            ValueError: If the slice is not an integer and max_ctx is not specified.
        """
        if self.mode == "int":
            return np.array([self.slice], dtype=np.int64)
        if max_ctx is None:
            raise ValueError("max_ctx must be specified if slice is not an integer")
        return np.arange(max_ctx, dtype=np.int64)[self.slice]

    def __repr__(
        self,
    ) -> str:
        return f"Slice: {self.slice} Mode: {self.mode} "
    


def get_act_name(
    name: str,
    layer: Optional[Union[int, str]] = None,
    layer_type: Optional[str] = None,
):
    """
    Helper function to convert shorthand to an activation name. Pretty hacky, intended to be useful for short feedback
    loop hacking stuff together, more so than writing good, readable code. But it is deterministic!

    Returns a name corresponding to an activation point in a TransformerLens model.

    Args:
         name (str): Takes in the name of the activation. This can be used to specify any activation name by itself.
         The code assumes the first sequence of digits passed to it (if any) is the layer number, and anything after
         that is the layer type.

         Given only a word and number, it leaves layer_type as is.
         Given only a word, it leaves layer and layer_type as is.

         Examples:
             get_act_name('embed') = get_act_name('embed', None, None)
             get_act_name('k6') = get_act_name('k', 6, None)
             get_act_name('scale4ln1') = get_act_name('scale', 4, 'ln1')

         layer (int, optional): Takes in the layer number. Used for activations that appear in every block.

         layer_type (string, optional): Used to distinguish between activations that appear multiple times in one block.

    Full Examples:

    get_act_name('k', 6, 'a')=='blocks.6.attn.hook_k'
    get_act_name('pre', 2)=='blocks.2.mlp.hook_pre'
    get_act_name('embed')=='hook_embed'
    get_act_name('normalized', 27, 'ln2')=='blocks.27.ln2.hook_normalized'
    get_act_name('k6')=='blocks.6.attn.hook_k'
    get_act_name('scale4ln1')=='blocks.4.ln1.hook_scale'
    get_act_name('pre5')=='blocks.5.mlp.hook_pre'
    """
    if (
        ("." in name or name.startswith("hook_"))
        and layer is None
        and layer_type is None
    ):
        # If this was called on a full name, just return it
        return name
    match = re.match(r"([a-z]+)(\d+)([a-z]?.*)", name)
    if match is not None:
        name, layer, layer_type = match.groups(0)

    layer_type_alias = {
        "a": "attn",
        "m": "mlp",
        "b": "",
        "block": "",
        "blocks": "",
        "attention": "attn",
    }

    act_name_alias = {
        "attn": "pattern",
        "attn_logits": "attn_scores",
        "key": "k",
        "query": "q",
        "value": "v",
        "mlp_pre": "pre",
        "mlp_mid": "mid",
        "mlp_post": "post",
    }

    layer_norm_names = ["scale", "normalized"]

    if name in act_name_alias:
        name = act_name_alias[name]

    full_act_name = ""
    if layer is not None:
        full_act_name += f"blocks.{layer}."
    if name in [
        "k",
        "v",
        "q",
        "z",
        "rot_k",
        "rot_q",
        "result",
        "pattern",
        "attn_scores",
    ]:
        layer_type = "attn"
    elif name in ["pre", "post", "mid", "pre_linear"]:
        layer_type = "mlp"
    elif layer_type in layer_type_alias:
        layer_type = layer_type_alias[layer_type]

    if layer_type:
        full_act_name += f"{layer_type}."
    full_act_name += f"hook_{name}"

    if name in layer_norm_names and layer is None:
        full_act_name = f"ln_final.{full_act_name}"
    return full_act_name

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
