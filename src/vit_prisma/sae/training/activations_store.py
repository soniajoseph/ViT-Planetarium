import os
from typing import Any, Iterator, cast

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from vit_prisma.models.base_vit import HookedViT

import numpy as np


def collate_fn(data):
    imgs = [d[0] for d in data]
    return torch.stack(imgs, dim=0)





def collate_fn_eval(data):
    imgs = [d[0] for d in data]
    return torch.stack(imgs, dim=0), torch.tensor([d[1] for d in data])



class VisionActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    def __init__(
        self,
        cfg: Any,
        model: HookedViT,
        dataset,
        create_dataloader: bool = True,
        eval_dataset=None,
        num_workers=0,
    ):
        self.cfg = cfg
        self.model = model
        self.model.to(cfg.device)
        self.dataset = dataset

        self.image_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=self.cfg.store_batch_size,
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.image_dataloader_eval = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=self.cfg.store_batch_size,
            collate_fn=collate_fn_eval,
            drop_last=True,
        )

        self.image_dataloader_iter = self.get_batch_tokens_internal()
        self.image_dataloader_eval_iter = self.get_val_batch_tokens_internal()

        if self.cfg.use_cached_activations:  # EDIT: load from multi-layer acts
            assert self.cfg.cached_activations_path is not None  # keep pyright happy
            # Sanity check: does the cache directory exist?
            assert os.path.exists(
                self.cfg.cached_activations_path
            ), f"Cache directory {self.cfg.cached_activations_path} does not exist. Consider double-checking your dataset, model, and hook names."

            self.next_cache_idx = 0  # which file to open next
            self.next_idx_within_buffer = 0  # where to start reading from in that file

            # Check that we have enough data on disk
            first_buffer = torch.load(f"{self.cfg.cached_activations_path}/0.pt")
            buffer_size_on_disk = first_buffer.shape[0]
            n_buffers_on_disk = len(os.listdir(self.cfg.cached_activations_path))
            # Note: we're assuming all files have the same number of tokens
            # (which seems reasonable imo since that's what our script does)
            n_activations_on_disk = buffer_size_on_disk * n_buffers_on_disk
            assert (
                n_activations_on_disk > self.cfg.total_training_tokens
            ), f"Only {n_activations_on_disk/1e6:.1f}M activations on disk, but cfg.total_training_tokens is {self.cfg.total_training_tokens/1e6:.1f}M."

            # TODO add support for "mixed loading" (ie use cache until you run out, then switch over to streaming from HF)

        if create_dataloader:
            # fill buffer half a buffer, so we can mix it with a new buffer
            self.storage_buffer = self.get_buffer(self.cfg.n_batches_in_buffer // 2)
            self.dataloader = self.get_data_loader()

    def get_batch_tokens_internal(self):
        """
        Streams a batch of tokens from a dataset.
        """
        # TODO could keep diff image sizes..

        # TODO multi worker here
        device = self.cfg.device
        # fetch a batch of images... (shouldn't this be it's own dataloader...)

        while True:
            for data in self.image_dataloader:
                data.requires_grad_(False)
                yield data.to(device)  # 5

    def get_batch_tokens(self):
        return next(self.image_dataloader_iter)

    # returns the ground truth class as well.
    def get_val_batch_tokens_internal(self):
        """
        Streams a batch of tokens from a dataset.
        """
        # TODO could keep diff image sizes..

        # TODO multi worker here
        device = self.cfg.device
        # fetch a batch of images... (shouldn't this be it's own dataloader...)
        # agreed, what the hell is going on here
        while True:
            for image_data, labels in self.image_dataloader_eval:
                image_data.requires_grad_(False)
                labels.requires_grad_(False)
                yield image_data.to(device), labels.to(device)

    def get_val_batch_tokens(self):
        return next(self.image_dataloader_eval_iter)


    # for live eval
    # this gets the same batch (first) from the eval dataloader each time
    def get_val_activations_one_batch(self):
        num_layers = (
            len(self.cfg.hook_point_layer)
            if isinstance(self.cfg.hook_point_layer, list)
            else 1
        )  # Number of hook points or layers
        for image_data, labels in self.image_dataloader_eval:
            image_data.requires_grad_(False)
            labels.requires_grad_(False)
            break
        # return tuple of (tokens, labels)
        return self.get_activations(image_data.to(self.cfg.device)), labels


    def get_activations(self, batch_tokens: torch.Tensor, get_loss: bool = False):
        """
        Returns activations with shape determined by config:
        - If cls_token and head_index: (batch, 1, num_layers, head_dim)
        - If cls_token only: (batch, 1, num_layers, d_model)
        - If head_index only: (batch, seq_len, num_layers, head_dim)
        - If neither: (batch, seq_len, num_layers, d_model)
        """
        layers = (
            self.cfg.hook_point_layer
            if isinstance(self.cfg.hook_point_layer, list)
            else [self.cfg.hook_point_layer]
        )
        act_names = [self.cfg.hook_point.format(layer=layer) for layer in layers]
        hook_point_max_layer = max(layers)

        layerwise_activations = self.model.run_with_cache(
            batch_tokens,
            names_filter=act_names,
            stop_at_layer=hook_point_max_layer + 1,
        )[1]

        activations_list = []
        for act_name in act_names:
            acts = layerwise_activations[act_name]
            if self.cfg.hook_point_head_index is not None:
                # If we're selecting specific head, do this first
                acts = acts[:, :, self.cfg.hook_point_head_index]
            if self.cfg.cls_token_only:
                # Then select CLS token if needed
                acts = acts[:, 0:1]
            activations_list.append(acts)

        return torch.stack(activations_list, dim=2)


    def get_buffer(self, n_batches_in_buffer: int):
        context_size = self.cfg.context_size
        batch_size = self.cfg.store_batch_size
        d_in = self.cfg.d_in
        total_size = batch_size * n_batches_in_buffer
        num_layers = (
            len(self.cfg.hook_point_layer)
            if isinstance(self.cfg.hook_point_layer, list)
            else 1
        )  # Number of hook points or layers

        # TODO does this still work (see above)
        if self.cfg.use_cached_activations:
            # Load the activations from disk
            buffer_size = total_size * context_size
            # Initialize an empty tensor with an additional dimension for layers
            new_buffer = torch.zeros(
                (buffer_size, num_layers, d_in),
                dtype=self.cfg.dtype,
                device=self.cfg.device,
            )
            n_tokens_filled = 0

            # Assume activations for different layers are stored separately and need to be combined
            while n_tokens_filled < buffer_size:
                if not os.path.exists(
                    f"{self.cfg.cached_activations_path}/{self.next_cache_idx}.pt"
                ):
                    print(
                        "\n\nWarning: Ran out of cached activation files earlier than expected."
                    )
                    print(
                        f"Expected to have {buffer_size} activations, but only found {n_tokens_filled}."
                    )
                    if buffer_size % self.cfg.total_training_tokens != 0:
                        print(
                            "This might just be a rounding error — your batch_size * n_batches_in_buffer * context_size is not divisible by your total_training_tokens"
                        )
                    print(f"Returning a buffer of size {n_tokens_filled} instead.")
                    print("\n\n")
                    new_buffer = new_buffer[:n_tokens_filled, ...]
                    return new_buffer

                activations = torch.load(
                    f"{self.cfg.cached_activations_path}/{self.next_cache_idx}.pt"
                )
                taking_subset_of_file = False
                if n_tokens_filled + activations.shape[0] > buffer_size:
                    activations = activations[: buffer_size - n_tokens_filled, ...]
                    taking_subset_of_file = True

                new_buffer[
                    n_tokens_filled : n_tokens_filled + activations.shape[0], ...
                ] = activations

                if taking_subset_of_file:
                    self.next_idx_within_buffer = activations.shape[0]
                else:
                    self.next_cache_idx += 1
                    self.next_idx_within_buffer = 0

                n_tokens_filled += activations.shape[0]

            return new_buffer

        refill_iterator = range(0, batch_size * n_batches_in_buffer, batch_size)
        # Initialize empty tensor buffer of the maximum required size with an additional dimension for layers
        new_buffer = torch.zeros(
            (total_size, context_size, num_layers, d_in),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )

        for refill_batch_idx_start in refill_iterator:
            refill_batch_tokens = self.get_batch_tokens()  ######
            refill_activations = self.get_activations(refill_batch_tokens)
            
            if self.cfg.use_patches_only:
                refill_activations = refill_activations[:, 1:, :, :]

            new_buffer[
              refill_batch_idx_start : refill_batch_idx_start + batch_size, ...
            ] = refill_activations

            # pbar.update(1)

        new_buffer = new_buffer.reshape(-1, num_layers, d_in)
        new_buffer = new_buffer[torch.randperm(new_buffer.shape[0])]

        return new_buffer

    def get_data_loader(
        self,
    ) -> Iterator[Any]:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """

        batch_size = self.cfg.train_batch_size

        # 1. # create new buffer by mixing stored and new buffer
        mixing_buffer = torch.cat(
            [
                self.get_buffer(self.cfg.n_batches_in_buffer // 2),
                self.storage_buffer,
            ],  ####
            dim=0,
        )

        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]
        # 2.  put 50 % in storage
        self.storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # 3. put other 50 % in a dataloader
        dataloader = iter(
            DataLoader(
                # TODO: seems like a typing bug?
                cast(Any, mixing_buffer[mixing_buffer.shape[0] // 2 :]),
                batch_size=batch_size,
                shuffle=True,
            )
        )

        return dataloader

    def next_batch(self):
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self.dataloader = self.get_data_loader() #### 97
            return next(self.dataloader)
