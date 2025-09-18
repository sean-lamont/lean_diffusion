# Conditional Diffusion based on The Diffusion Duality
This project extends the approach outlined in the paper [The Diffusion Duality (ICML 2025)](https://arxiv.org/abs/2506.10892v1) to support conditional text generation tasks.
We focus on theorem proving using the LeanDojo dataset, enabling the model to generate proofs conditioned on given mathematical goals, 
and test with two different backbones (ModernBERT and DiffuCoder), although the framework is adaptable to other tasks and models.

The implementation allows for conditional generation by:

- Maintaining prompt/context tokens fixed during the diffusion process
- Ensuring the training and inference processes denoise only the target portion of the sequence

The file `conditional_diffusion.py` contains the core logic for conditional diffusion, with `trainer_base.py` being adapted to handle new backbones.

To extend the approach to other conditional generation tasks, modify `dataloader.py` to create appropriate datasets following the
structure of the `leandojo` dataset, and add any new backbones in `trainer_base.py` following the examples of ModernBERT and Diffucoder.

The config files in the `configs` directory can be adjusted to change hyperparameters, model settings, and dataset paths,
and follow the structure of the original project. 

## Installation 
Install the required packages using conda from the provided environment file:
```bash conda env create -f environment.yaml```

## Usage
Unpack the data caches:

```bash unzip bert_data_cache.zip```

```bash unzip diffucoder_data_cache.zip```

To train with ModernBERT as the backbone:

```bash scripts/train_leandojo_duo.sh```

To train with DiffuCoder as the backbone:

```bash scripts/train_leandojo_diffucoder.sh```
