from vit_prisma.utils.data_utils.imagenet_dict import IMAGENET_DICT

import os


def imagenet_index_from_word(search_term: str) -> int:
    """
    Finds the ImageNet index corresponding to a search term.

    Args:
        search_term (str): The search term to look up in the ImageNet dictionary.

    Returns:
        int: The index corresponding to the search term in the ImageNet dictionary.

    Raises:
        ValueError: If the search term is not found in the ImageNet dictionary.
    """

    # Convert the search term to lowercase to ensure case-insensitive matching
    search_term = search_term.lower()

    # Iterate over the dictionary and search for the term
    for key, value in IMAGENET_DICT.items():
        if (
            search_term in value.lower()
        ):  # Convert each value to lowercase for case-insensitive comparison
            return key  # Return the key directly once found

    # If the loop completes without returning, the term was not found; raise an exception
    raise ValueError(f"'{search_term}' not found in IMAGENET_DICT.")


def setup_imagenet_paths(imagenet_path, format="kaggle") -> dict:
    # Currently based on kaggle dataset structure, can be modified for particular dataset type.

    if format == "kaggle":
        return {
            "train": os.path.join(imagenet_path, "ILSVRC/Data/CLS-LOC/train"),
            "val": os.path.join(imagenet_path, "ILSVRC/Data/CLS-LOC/val"),
            "val_labels": os.path.join(imagenet_path, "LOC_val_solution.csv"),
            "label_strings": os.path.join(imagenet_path, "LOC_synset_mapping.txt"),
        }

    elif format == "hhi":
        return {
            "train": os.path.join(imagenet_path, "train"),
            "val": os.path.join(imagenet_path, "val"),
            "val_labels": os.path.join(imagenet_path, "LOC_val_solution.csv"),
            "label_strings": os.path.join(imagenet_path, "LOC_synset_mapping.txt"),
        }
