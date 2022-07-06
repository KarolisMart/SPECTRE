# SPECTRE
Reference implementation for [SPECTRE: Spectral Conditioning Helps to Overcome the Expressivity Limits of One-shot Graph Generators (ICML 2022)](https://arxiv.org/abs/2204.01613)

## Setup

To run the code install all the necessary packages via the conda environment named `SPECTRE`:
```
conda env create -f environment.yml
```
 
## Code structure
`full_gan.py` holds the main PyTorch Lightning model

`model` holds individual generators, discriminators, and their internal components

`test.py` tests the model. After training a model supply the appropriate checkpoint to this script. It also accepts `--EMD` flag to compute the MMDs using earth movers distance following GraphRNN instead of the much faster Gaussian TV kernel used by GRAN.

`data.py` holds all of the code required to build and load the datasets we used. Our generated synthetic datasets can be found in the `data` folder. If you use these datasets in your code, you can use code similar to the function below, to get matching splits:
```
def load_graphs(filename, data_dir='data'):
  adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(f'{data_dir}/{filename}.pt')
  print(f'Dataset {filename} loaded from file')
  
  test_len = int(round(len(adjs)*0.2))
  train_len = int(round((len(adjs) - test_len)*0.8))
  val_len = len(adjs) - train_len - test_len
  print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
  
  train, val, test = torch.utils.data.random_split(graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))

  return train, val, test
```

If orbital count MMD does not work (is always zero) you might need to recompile the `util/orca/orca` executable and when used on Linux make sure that the appropriate user has permission to execute it.

## Output files
Tensorboard logs, hyperparameters used and model checkpoints are saved in the `logs` folder. Running `test.py` on a trained model creates a `test` subfolder for the run with six more folders, each of which holds generated graphs in a `generated_graphs.pt` file generated when conditioning on nothing, true eigenvalues or true eigenvalues or eigenvectors. This is repeated for the exponential moving average (EMA) model weights.

## Example commands used to train our models

Planar:
```
python full_gan.py --batch_size 10 --max_epochs=12000 --log_every_n_steps=60 --check_val_every_n_epoch 90 --gen_gelu --disc_gelu --k_eigval 2 --eigvec_temp_decay --decay_eigvec_temp_over 2000 --min_eigvec_temp 0.8 --n_eigval_warmup_epochs 2000 --n_eigvec_warmup_epochs 2000 --eigval_temp_decay --decay_eigval_temp_over 2000 --min_eigval_temp 0.8 --SON_D_full_readout --noisy_gen --SON_normalize_left --lambda_gating --lambda_last_gating --lambda_upsample --SON_small --noisy_disc --derived_eigval_noise --normalize_noise --spectral_norm --eigvec_right_noise --gp_shared_alpha --no_restart --SON_gumbel_temperature_decay --decay_SON_gumbel_temp_over 10000 --SON_gumbel_temperature_warmup_epochs 0 --n_data_workers 4 --dataset planar --n_nodes 64 --gp_do_backwards --eigvec_sign_flip --ignore_first_eigv --gp_include_unpermuted --clip_grad_norm 1.0 --seed 0
```

SBM:
```
python full_gan.py --batch_size 5 --max_epochs=6000 --log_every_n_steps=60 --check_val_every_n_epoch 45 --gen_gelu --disc_gelu --k_eigval 4 --eigvec_temp_decay --decay_eigvec_temp_over 1000 --min_eigvec_temp 0.8 --n_eigval_warmup_epochs 1000 --n_eigvec_warmup_epochs 1000 --eigval_temp_decay --decay_eigval_temp_over 1000 --min_eigval_temp 0.8 --SON_D_full_readout --noisy_gen --SON_normalize_left --lambda_gating --lambda_last_gating --lambda_upsample --SON_small --noisy_disc --derived_eigval_noise --normalize_noise --spectral_norm --eigvec_right_noise --gp_shared_alpha --no_restart --SON_gumbel_temperature_decay --decay_SON_gumbel_temp_over 5000 --SON_gumbel_temperature_warmup_epochs 0 --n_data_workers 4 --dataset sbm --gp_do_backwards --eigvec_sign_flip --ignore_first_eigv --gp_include_unpermuted --clip_grad_norm 1.0
```

Proteins:
```
python full_gan.py --batch_size 1 --max_epochs=1020 --log_every_n_steps=60 --check_val_every_n_epoch 20 --gen_gelu --disc_gelu --k_eigval 16 --eigvec_temp_decay --decay_eigvec_temp_over 176 --min_eigvec_temp 0.8 --n_eigval_warmup_epochs 176 --n_eigvec_warmup_epochs 176 --eigval_temp_decay --decay_eigval_temp_over 176 --min_eigval_temp 0.8 --SON_D_full_readout --noisy_gen --SON_normalize_left --lambda_gating --lambda_last_gating --lambda_upsample --SON_small --noisy_disc --derived_eigval_noise --normalize_noise --spectral_norm --eigvec_right_noise --gp_shared_alpha --no_restart --SON_gumbel_temperature_decay --decay_SON_gumbel_temp_over 875 --SON_gumbel_temperature_warmup_epochs 0 --n_data_workers 4 --dataset protein --gp_do_backwards --eigvec_sign_flip --ignore_first_eigv --gp_include_unpermuted --clip_grad_norm 1.0 --accelerator 'ddp' --gpus 4 --seed 0
```

Community:
```
python full_gan.py --batch_size 10 --max_epochs=12000 --log_every_n_steps=60 --check_val_every_n_epoch 90 --gen_gelu --disc_gelu --k_eigval 2 --eigvec_temp_decay --decay_eigvec_temp_over 2000 --min_eigvec_temp 0.8 --n_eigval_warmup_epochs 2000 --n_eigvec_warmup_epochs 2000 --eigval_temp_decay --decay_eigval_temp_over 2000 --min_eigval_temp 0.8 --SON_D_full_readout --noisy_gen --SON_normalize_left --lambda_gating --lambda_last_gating --lambda_upsample --SON_small --noisy_disc --derived_eigval_noise --normalize_noise --spectral_norm --eigvec_right_noise --gp_shared_alpha --no_restart --SON_gumbel_temperature_decay --decay_SON_gumbel_temp_over 10000 --SON_gumbel_temperature_warmup_epochs 0 --n_data_workers 4 --dataset community --n_start 12 --n_end 21 --n_graphs 100 --gp_do_backwards --eigvec_sign_flip --ignore_first_eigv --gp_include_unpermuted --clip_grad_norm 1.0
```

QM9 (following MolGAN):
```
python full_gan.py --batch_size 128 --max_epochs=30 --log_every_n_steps=80 --check_val_every_n_epoch 1 --gen_gelu --disc_gelu --n_G 3 --n_D 3 --k_eigval 2 --eigvec_temp_decay --decay_eigvec_temp_over 10 --min_eigvec_temp 0.8 --n_eigval_warmup_epochs 10 --n_eigvec_warmup_epochs 15 --eigval_temp_decay --decay_eigval_temp_over 10 --min_eigval_temp 0.8 --SON_D_full_readout --noisy_gen --SON_normalize_left --lambda_gating --lambda_last_gating --lambda_upsample --SON_small --noisy_disc --derived_eigval_noise --normalize_noise --spectral_norm --eigvec_right_noise --gp_shared_alpha --no_restart --SON_gumbel_temperature_decay --decay_SON_gumbel_temp_over 25 --SON_gumbel_temperature_warmup_epochs 0 --n_data_workers 4 --dataset qm9 --n_graphs -1 --gp_do_backwards --eigvec_sign_flip --ignore_first_eigv --gp_include_unpermuted --clip_grad_norm 1.0
```
Adding `--qm9_strict_eval` flag during evaluation or training (for model selection) only counts molecules as valid if they have one connected component.

To train MolGAN* add `--mlp_gen`,
to train GG-GAN* add `--use_fixed_emb`,
to train GG-GAN (RS)* add `--no_cond`.
These baselines only generate adjacencies, like SPECTRE trained with `--adj_only` flag.

The code has been simplified and refactored, so the results might slightly differ from the published ones.

The PPGN GAN can sometimes get stuck. If you encounter stability issues when training the model on your problem, you can try increasing the number of eigenvectors considered (`--k_eigval`), increase the permutations used for the gradient penalty (`--gp_adj_rewire` and `--gp_adj_noise`), use the (`--cat_mult_eigvals`) option or try a different random seed (`--seed`) as some can be unlucky.
