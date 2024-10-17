# SAE Experiment Logging and Tracking

An objective we have when training an SAE is to have a small number of non-zero elements in the hidden layer (i.e. a small L0 'norm'). The L1 norm is used as a proxy for this.

The loss term of the SAE is:

$$loss = mse + l1\_coeff * sparsity + [ \: ghost \: gadients \: ]$$

The two loss term in the SAE loss function have conflicting goals:
* The **reconstruction term** (MSE) works to make the autoencoder good at reconstructing the input.
* The **sparsity** term works to reduce the magnitudes in the hidden layer. 

When training an SAE most metrics are related to measuring reconstruction loss vs sparsity, in different ways. The goal is to monitor these and find pareto improvements.

Weights and Biases is used to track experiments (toggled with `VisionModelSAERunnerConfig.log_to_wandb`)

# Validation

Whilst training an SAE validation metrics are logged every `VisionModelSAERunnerConfig.wandb_log_frequency` steps, whilst the sparsity is logged every `VisionModelSAERunnerConfig.feature_sampling_window` steps. Note that each step is `VisionModelSAERunnerConfig.train_batch_size` number of tokens, for a total of (num_images x num_epochs x tokens_per_image / training_batch) steps.

During validation the following matrics are logged:

#### plots

* `log_feature_density_histogram` - every step since the last log the number of feature activations that are greater than 0 divided by the total amount. See [here](https://arena3-chapter1-transformer-interp.streamlit.app/[1.3.2]_Interpretability_with_SAEs) for a discussion on how to interpret this. 

#### details

* `details/current_learning_rate` - The learning rate at each step of training.
* `details/n_training_tokens` - The number of training tokens seen so far (Images x tokens per image)
* `details/n_training_images` - The number of training images seen so far

#### losses

* `losses/mse_loss` - The normalised MSE loss between the SAE input and output, i.e. the reconstruction loss. Normalised such that it is less dependent on the size of the input/output. 
* `losses/l1_loss`
* `losses/ghost_grad_loss` - this describes the method of adding an additional term to the loss, which essentially gives dead latents a gradient signal that pushes them in the direction of explaining more of the autoencoder's residual.
`losses/overall_loss`

#### metrics


* `metrics/explained_variance` - explained variance is the ratio of the variance in the reconstructed data to the variance in the original input data. A higher explained variance indicates that the SAE has learned features that capture more of the meaningful structure in the data. This is the mean over the batch.
* `metrics/explained_variance_std` - This is the std over the batch.
* `metrics/l0` - the feature activations for that step greater than 0 summed together, and the mean over the batch.
* `metrics/mean_log10_feature_sparsity`

#### sparsity

* `sparsity/mean_passes_since_fired` - For each activation, Tracks the number of training tokens (images x tokens per image) since activation was greater than 0
* `sparsity/dead_features`
* `sparsity/below_1e-5` - feature_sparsity < 1e-5
* `sparsity/below_1e-6` - feature_sparsity < 1e-6

# Evaluation

There are two types of evaluation, namely that which is called during training and after training. This is because it is a costly procedure and the `TrainingEvalConfig` defines an evaluation run that is performed multiple times during training to give an idea of SAE performance and `PostTrainingEvalConfig` which aims to give a full breakdown of model performance. 

## Training Evaluation

Whilst training an SAE the training evaluation will run every `EvalConfig.log_frequency` and a post-training evaluation will run at the end. A training run can be kicked off with: 

TODO EdS: Document which evals are available

```python
cfg = VisionModelSAERunnerConfig()
trainer = VisionSAETrainer(cfg)
sae = trainer.run()
```

## Post-Training Evaluation

To manually kick off a post-training evaluation on a saved SAE:

```python
model = load_model(cfg)
sae = load_sae(cfg)
train_data, val_data, val_data_visualize = load_dataset(cfg)
evaluator = Evaluator(model, val_data, cfg, visualize_data=val_data_visualize)
evaluator.evaluate(sae, context="post-training")
```

TODO EdS: CLI implementation for evaluation

