# Usage Guide

The Prisma Project is a framework that facilitates training Vision Transformers (ViTs) in various configurations, with synthetically generated datasets suited for mechanistic interpretability. The library also allows loading pretrained models from huggingface and timm, which can further be finetuned using the trainer built into the framework.

## Installing Repo

To install as an editable repo:

```
git clone https://github.com/soniajoseph/ViT-Prisma
cd ViT-Prisma
pip install -e .
```

## Configurations 

The Prisma config object houses certain hyperparameters of the models, along with attributes of the training procedure, which can come in handy to track, and design new experiments. 

```python
from vit_prisma.configs.HookedViTConfig import HookedViTConfig
```
By inspecting the HookedViTConfig config available in the framework, you can create your own one based on the experiment.

## Hooked ViT

The Hooked ViT part of the Prisma Project allows instantiating models in a flexible way along with hooks to inspect the internals of a model during forward / backward passes, making it easier for mechanistic interpretability research with ViTs. The architecture is defined by the Prisma config object discussed in the previous section.

### Instantiating the Base ViT 

```python
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.configs.HookedViTConfig import HookedViTConfig

# The HookedViTConfig is available in the framework itself, which is the base setup for any experiment, one can customize it as per the requirements of an experiment.
config = HookedViTConfig()

model = HookedViT(config)
```

### Using pretrained models

```python
from vit_prisma.models.pretrained_model import PretrainedModel
from vit_prisma.configs.HookedViTConfig import HookedViTConfig

config = HookedViTConfig()

hf_model = PretrainedModel('google/vit-base-patch16-224-in21k', config)

timm_model = PretrainedModel('vit_base_patch32_224', config, is_timm=True)
```

### Datasets

The framework also has a few datasets that can be synthetically generated, and cached. 

Below are the datasets available as part of the framework: 
 - Circle
 - [dsprites](https://github.com/google-deepmind/dsprites-dataset)
 - Induction

You will be able to use any other dataset with the framework, for ease of use, you can instantiate it as a Pytorch dataset object.

 ```python
 from vit_prisma.dataloaders.induction import InductionDataset

 train = InductionDataset('train')

 # If you don't provide a test split to the trainer, the trainer will perform the train/test split for you.
 ```

 ### Training and tracking experiments

 The trainer has built in support for wandb experiment tracking. Make sure to set up wandb on your localhost, and configure the tracking attributes in the Prisma config object.

 ```python
 from vit_prisma.models.base_vit import HookedViT
 from vit_prisma.configs.HookedViTConfig import HookedViTConfig
 from vit_prisma.training import trainer
 from vit_prisma.dataloaders.induction import InductionDataset

 train_dataset = InductionDataset('train')

 config = HookedViTConfig()

 model_function = HookedViT

 trainer.train(model_function, config, train_dataset)
 ```

 ### Callbacks

The trainer has support for callbacks, that allows you to pass methods that can be executed after a desired number of epoch(s), or step(s). To setup callback methods, make sure to follow the PrismaCallback protocol.

 ```python
 from vit_prisma.models.base_vit import HookedViT
 from vit_prisma.configs.HookedViTConfig import HookedViTConfig
 from vit_prisma.training import trainer
 from vit_prisma.dataloaders.induction import InductionDataset

 from vit_prisma.training.training_utils import PrismaCallback

 class DemoCallback(PrismaCallback):
    def on_epoch_end(self, epoch, net, val_loader, wandb_logger):
        # Specify a condition if you don't want the callback to execute after each epoch
        if epoch % 5 == 0: 
            # perform some function with the network, validation set, and log it if required.
            pass

    def on_step_end(self, step, net, val_loader, wandb_logger):
        # It is similar to on_epoch_end but runs after desired number of steps instead of epochs
        pass

 train_dataset = InductionDataset('train')

 config = HookedViTConfig()

 model_function = HookedViT

 trainer.train(model_function, config, train_dataset, callbacks=[DemoCallback()])
 ```

## Git Hooks

We have git hooks that run `ruff` (linter), `black` (code formatter), and `isort` (standardising imports), to improve maintainability of the code base.

To set these up all you need to do is run:

```
pre-commit install 
```

Now every time you try to make a commit, the hooks will run automatically on your staged files.

Further instructions:

* To manually run the hooks on all files: `pre-commit run --all-files`
* To manually run on staged files only: `pre-commit run`
* To skip (not recommended) for a commit: `git commit -m "your message" --no-verify`
