from argparse import ArgumentParser
import os
import time
import random
import string
import math
from sklearn.cluster import KMeans
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from typing import Tuple
from torch_ema import ExponentialMovingAverage

from util.model_helper import zero_diag, rand_rot, rand_rewire, eigval_noise, deterministic_vector_sign_flip, sort_eigvecs, interpolate_eigvecs, reorder_adj, categorical_permute
from util.eval_helper import degree_stats, orbit_stats_all, clustering_stats, spectral_stats, eigval_stats, spectral_filter_stats, eval_acc_lobster_graph, eval_acc_tree_graph, eval_acc_grid_graph, eval_acc_sbm_graph, eval_acc_planar_graph, eval_fraction_isomorphic, eval_fraction_unique_non_isomorphic_valid, eval_fraction_unique, compute_list_eigh, is_lobster_graph, is_grid_graph, is_sbm_graph, is_planar_graph
from model.noise_mlp import NoiseMLP
from model.lambda_gan import LambdaDiscriminator, LambdaGenerator
from model.SON_gan import SONPointNetDiscriminator, SONGenerator
from model.ppgn_gan import PPGNDiscriminator, PPGNGenerator, MLPGenerator
from data import GraphDataModule, N_MAX

class GradMonitor(pl.callbacks.Callback):
    """
    Callbacks to log param and their gradient histograms
    """

    def on_after_backward(self, trainer, pl_module):
        global_step = trainer.global_step
        # Respect loging frequency
        if ((global_step + 1) % trainer.log_every_n_steps == 0 or trainer.should_stop):
            for name, param in trainer.model.named_parameters():
                name = name.split('.')
                trainer.logger.experiment.add_histogram(f"{name[0]}/{'.'.join(name[1:])}", param, global_step)
                if param.requires_grad and param.grad is not None:
                    trainer.logger.experiment.add_histogram(f"{name[0]}_grad/{'.'.join(name[1:])}", param.grad, global_step)

class SPECTRE(pl.LightningModule):
    """
    Spectral graph GAN (WGAN-LP)
    """

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--seed", default=1234, type=int)
        parser.add_argument("--beta1", default=0.5, type=float)
        parser.add_argument("--beta2", default=0.9, type=float)
        parser.add_argument("--lr_g", default=1e-4, type=float)
        parser.add_argument("--lr_d", default=1e-4, type=float)
        parser.add_argument('--gp_lambda', default=5, type=float)
        parser.add_argument("--n_max", default=N_MAX, type=int)
        parser.add_argument("--gen_leaky_ReLU_alpha", default=0.0, type=float)
        parser.add_argument("--disc_leaky_ReLU_alpha", default=0.2, type=float)
        parser.add_argument('--n_G', default=8, type=int)
        parser.add_argument('--n_D', default=8, type=int)
        parser.add_argument('--hid_G', default=64, type=int)
        parser.add_argument('--hid_D', default=64, type=int)
        parser.add_argument('--disc_normalization', default='instance', type=str)
        parser.add_argument('--gen_normalization', default='instance', type=str)
        parser.add_argument('--k_eigval', default=2, type=int)
        parser.add_argument('--SON_shared_weights', default=False, action="store_true")
        parser.add_argument('--gen_gelu', default=False, action="store_true")
        parser.add_argument('--disc_gelu', default=False, action="store_true")
        parser.add_argument('--disc_step_multiplier', default=1, type=int)
        parser.add_argument('--weight_decay', default=0.0, type=float)
        parser.add_argument('--use_fixed_emb', default=False, action="store_true")
        parser.add_argument('--eigvec_temp_decay', default=False, action="store_true")
        parser.add_argument('--decay_eigvec_temp_over', default=10, type=int)
        parser.add_argument('--spectral_norm', default=False, action="store_true")
        parser.add_argument('--G_dropout', default=0.1, type=float)
        parser.add_argument('--D_dropout', default=0.1, type=float)
        parser.add_argument('--skip_connection', default=False, action="store_true")
        parser.add_argument('--n_rot', default=3, type=int)
        parser.add_argument('--cat_eigvals', default=False, action="store_true")
        parser.add_argument('--cat_mult_eigvals', default=False, action="store_true")
        parser.add_argument('--disc_aux', default=0.0, type=float)
        parser.add_argument('--n_eigval_warmup_epochs', default=0, type=int)
        parser.add_argument('--n_eigvec_warmup_epochs', default=0, type=int)
        parser.add_argument('--SON_disc', default=1.0, type=float)
        parser.add_argument('--eigval_noise', default=0.01, type=float)
        parser.add_argument('--min_eigvec_temp', default=0.0, type=float)
        parser.add_argument('--SON_max_pool', default=False, action="store_true")
        parser.add_argument('--SON_skip_connection', default=False, action="store_true")
        parser.add_argument('--SON_share_weights', default=False, action="store_true")
        parser.add_argument('--SON_D_full_readout', default=False, action="store_true")
        parser.add_argument('--SON_D_n_rot', default=2, type=int)
        parser.add_argument('--rand_rot_var', default=0.2, type=float)
        parser.add_argument('--noise_latent_dim', default=100, type=int)
        parser.add_argument('--lambda_disc', default=1.0, type=float)
        parser.add_argument('--eigval_temp_decay', default=False, action="store_true")
        parser.add_argument('--decay_eigval_temp_over', default=10, type=int)
        parser.add_argument('--min_eigval_temp', default=0.0, type=float)
        parser.add_argument('--max_eigval_temp', default=1.0, type=float)
        parser.add_argument('--max_eigvec_temp', default=1.0, type=float)
        parser.add_argument('--adj_noise', default=0.005, type=float)
        parser.add_argument('--eigvec_noise', default=0.005, type=float)
        parser.add_argument('--edge_noise', default=False, action="store_true")
        parser.add_argument('--edge_eigvecs', default=False, action="store_true")
        parser.add_argument('--lambda_only', default=False, action="store_true")
        parser.add_argument('--lambda_norm', default='instance', type=str)
        parser.add_argument('--lambda_upsample', default=False, action="store_true")
        parser.add_argument('--lr_decay_every', default=10, type=int)
        parser.add_argument('--lr_decay_warmup', default=10, type=int)
        parser.add_argument('--lr_D_decay', default=1.0, type=float)
        parser.add_argument('--lr_G_decay', default=1.0, type=float)
        parser.add_argument('--adj_only', default=False, action="store_true")
        parser.add_argument('--adj_eigvec_only', default=False, action="store_true")
        parser.add_argument('--SON_only', default=False, action="store_true")
        parser.add_argument('--lambda_SON_only', default=False, action="store_true")
        parser.add_argument('--SON_normalize_left', default=False, action="store_true")
        parser.add_argument('--noisy_gen', default=False, action="store_true")
        parser.add_argument('--lambda_gating', default=False, action="store_true")
        parser.add_argument('--lambda_last_gating', default=False, action="store_true")
        parser.add_argument('--lambda_last_linear', default=False, action="store_true")
        parser.add_argument('--lambda_dropout', default=0.0, type=float)
        parser.add_argument('--gp_adj_rewire', default=0.1, type=float)
        parser.add_argument('--gp_adj_noise', default=0.05, type=float)
        parser.add_argument('--wgan_eps', default=1e-3, type=float)
        parser.add_argument('--ema', default=0.995, type=float)
        parser.add_argument('--compute_emd', default=False, action="store_true")
        parser.add_argument('--noisy_disc', default=False, action="store_true")
        parser.add_argument('--SON_small', default=False, action="store_true")
        parser.add_argument('--temp_new', default=False, action="store_true")
        parser.add_argument('--pretrain', default=0, type=int)
        parser.add_argument('--gp_do_backwards', default=False, action="store_true")
        parser.add_argument('--disc_noise_rewire', default=0.05, type=float)
        parser.add_argument('--D_partial_laplacian', default=False, action="store_true")
        parser.add_argument('--derived_eigval_noise', default=False, action="store_true")
        parser.add_argument('--normalize_noise', default=False, action="store_true")
        parser.add_argument('--SON_init_bank_size', default=10, type=int)
        parser.add_argument('--SON_gumbel_temperature', default=1.0, type=float)
        parser.add_argument('--eigvec_right_noise', default=False, action="store_true")
        parser.add_argument('--min_SON_gumbel_temperature', default=0.0625, type=float)
        parser.add_argument('--SON_gumbel_temperature_decay', default=False, action="store_true")
        parser.add_argument('--decay_SON_gumbel_temp_over', default=10, type=int)
        parser.add_argument('--SON_gumbel_temperature_warmup_epochs', default=0, type=int)
        parser.add_argument('--gp_shared_alpha', default=False, action="store_true")
        parser.add_argument('--sharp_restart', default=False, action="store_true")
        parser.add_argument('--no_restart', default=False, action="store_true")
        parser.add_argument('--precise_uniqueness_val', default=False, action="store_true")
        parser.add_argument('--SON_kl_init_scale', default=5e-4, type=float)
        parser.add_argument('--SON_stiefel_sim_init', default=False, action="store_true")
        parser.add_argument('--mlp_gen', default=False, action="store_true")
        parser.add_argument('--use_eigvecs', default=False, action="store_true")
        parser.add_argument('--no_cond', default=False, action="store_true")
        parser.add_argument('--init_emb_channels', default=64, type=int)
        parser.add_argument('--eigvec_sign_flip', default=False, action="store_true")
        parser.add_argument('--gp_include_unpermuted', default=False, action="store_true")
        parser.add_argument('--ppgn_data_channels_mult',  default=1, type=int)
        parser.add_argument('--skip_noise_preprocess', default=False, action="store_true")
        parser.add_argument('--clip_grad_norm', default=-1.0, type=float)
        
        return parser

    def __init__(
        self, beta1: float = 0.5, beta2: float = 0.9, lr_g: float = 1e-4, lr_d: float = 3e-4, gp_lambda: float = 10, n_max: int = N_MAX, gen_leaky_ReLU_alpha: float = 0.2,
        disc_leaky_ReLU_alpha: float = 0.2, lp_penalty: bool = False, n_G: int = 8, n_D: int = 8, hid_G: int = 64, hid_D: int = 64, disc_normalization: str = 'instance',
        gen_gelu: bool = False, disc_gelu: bool = False, k_eigval: int = 18, ppgn: bool = False, smp: bool = False, disc_step_multiplier: int = 1, weight_decay: float = 0.0,
        use_fixed_emb: bool=False, gen_normalization: str = 'instance', eigvec_temp_decay: bool = False, decay_eigvec_temp_over: float = 0.0, spectral_norm: bool = False,
        G_dropout: float = 0.0, D_dropout: float = 0.0, skip_connection: bool = False, n_rot: int = 1, cat_eigvals: bool = False, cat_mult_eigvals: bool = False,
        disc_aux: float = 0.0, n_eigval_warmup_epochs: int = 0, n_eigvec_warmup_epochs: int = 0, SON_disc: float = 1.0, eigval_noise: float = 0.01, min_eigvec_temp: float = 0.0,
        SON_max_pool: bool = False, SON_skip_connection: bool = False, SON_share_weights: bool = False, SON_D_full_readout: bool = False, SON_D_n_rot: int = 2,
        rand_rot_var: float = 0.1, noise_latent_dim: int = 100, lambda_disc: float = 1.0, eigval_temp_decay: bool = False, decay_eigval_temp_over: float = 10,
        min_eigval_temp: float = 0.0, max_eigval_temp: float = 1.0, max_eigvec_temp: float = 1.0, adj_noise: float = 0.005, eigvec_noise: float = 0.005, edge_noise: bool = False,
        edge_eigvecs: bool = False, lambda_only: bool = False, lambda_norm: str = 'instance', lambda_upsample: bool = False, lr_decay_every: int = 10, lr_decay_warmup: int = 2000,
        lr_D_decay: float = 1.0, lr_G_decay: float = 1.0, adj_only: bool = False, adj_eigvec_only: bool = False, SON_only: bool = False, lambda_SON_only: bool = False,
        SON_normalize_left: bool = False, noisy_gen: bool = False, lambda_gating: bool = False, lambda_last_gating: bool = False, lambda_last_linear: bool = False,
        lambda_dropout: float = 0.0, gp_adj_rewire: float = 0.1, gp_adj_noise: float = 0.05, wgan_eps: float = 0.0,
        ema: float = 0.995, compute_emd: bool = False, noisy_disc: bool = False, SON_small: bool = False, temp_new: bool = False, pretrain: int = 0, gp_do_backwards: bool = False,
        disc_noise_rewire: float = 0.05, D_partial_laplacian: bool = False, derived_eigval_noise: bool = False, normalize_noise: bool = False, SON_init_bank_size: int = 10,
        SON_gumbel_temperature: float = 1.0, eigvec_right_noise: bool = False, min_SON_gumbel_temperature: float = 0.0625, SON_gumbel_temperature_decay: bool = False,
        decay_SON_gumbel_temp_over: int = 10, SON_gumbel_temperature_warmup_epochs: int = 0, gp_shared_alpha: bool = False, sharp_restart: bool = False,
        no_restart: bool = False, precise_uniqueness_val: bool = False, SON_kl_init_scale: float = 5e-4, SON_stiefel_sim_init: bool = False, mlp_gen: bool = False,
        use_eigvecs: bool = False, no_cond: bool = False, init_emb_channels: int = 64, eigvec_sign_flip: bool = False, ignore_first_eigv: bool = False, 
        gp_include_unpermuted: bool = False, ppgn_data_channels_mult: int = 1.0, skip_noise_preprocess: bool = False, clip_grad_norm: float = -1.0, qm9: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Generators.
        self.generator = self._get_generator()
        self.SON_generator = self._get_SON_generator()
        self.lambda_generator = self._get_lambda_generator()
        # Discriminators
        self.discriminator = self._get_discriminator()
        self.SON_discriminator = self._get_SON_discriminator()
        self.lambda_discriminator = self._get_lambda_discriminator()
        # Noise pre-processing MLPs
        self.mlp_noise, self.mlp_SON_node_noise, self.mlp_adj_node_noise = self._get_noise_mlps()

        self.gen_params = list(self.generator.parameters()) + list(self.SON_generator.parameters()) + list(self.lambda_generator.parameters()) + list(self.mlp_noise.parameters()) + list(self.mlp_SON_node_noise.parameters()) + list(self.mlp_adj_node_noise.parameters())

        self.automatic_optimization = False # Do optimizer step manually to lower memory consumption for GP

        # Track generators exponential moving average
        self.gen_ema = ExponentialMovingAverage(self.gen_params, decay=self.hparams.ema)

    def _get_noise_mlps(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        mlp_noise = NoiseMLP(input_noise_latent_dim=self.hparams.noise_latent_dim+1, output_noise_latent_dim=self.hparams.noise_latent_dim, alpha=self.hparams.gen_leaky_ReLU_alpha, gelu=self.hparams.gen_gelu)
        mlp_SON_node_noise = NoiseMLP(input_noise_latent_dim=self.hparams.noise_latent_dim+1, output_noise_latent_dim=self.hparams.noise_latent_dim, alpha=self.hparams.gen_leaky_ReLU_alpha, gelu=self.hparams.gen_gelu)
        mlp_adj_node_noise = NoiseMLP(input_noise_latent_dim=self.hparams.noise_latent_dim+1, output_noise_latent_dim=self.hparams.noise_latent_dim, alpha=self.hparams.gen_leaky_ReLU_alpha, gelu=self.hparams.gen_gelu)
        return mlp_noise, mlp_SON_node_noise, mlp_adj_node_noise

    def _get_generator(self) -> nn.Module:
        if self.hparams.mlp_gen:
            generator = MLPGenerator(alpha=self.hparams.gen_leaky_ReLU_alpha, n_max=self.hparams.n_max, noise_latent_dim=self.hparams.noise_latent_dim,
                        n_layers=self.hparams.n_G, data_channels=self.hparams.hid_G, gelu=self.hparams.gen_gelu,
                        k_eigval=self.hparams.k_eigval,
                        use_fixed_emb=self.hparams.use_fixed_emb,
                        skip_connection=self.hparams.skip_connection,
                        cat_eigvals=self.hparams.cat_eigvals, cat_mult_eigvals=self.hparams.cat_mult_eigvals, no_extra_n=(not self.hparams.skip_noise_preprocess),
                        use_eigvecs=self.hparams.use_eigvecs, qm9=self.hparams.qm9, data_channels_mult=self.hparams.ppgn_data_channels_mult)
        else:
            generator = PPGNGenerator(alpha=self.hparams.gen_leaky_ReLU_alpha, n_max=self.hparams.n_max, noise_latent_dim=self.hparams.noise_latent_dim,
                            n_layers=self.hparams.n_G, data_channels=self.hparams.hid_G, gelu=self.hparams.gen_gelu,
                            k_eigval=self.hparams.k_eigval,
                            use_fixed_emb=self.hparams.use_fixed_emb, normalization=self.hparams.gen_normalization,
                            dropout=self.hparams.G_dropout,
                            skip_connection=self.hparams.skip_connection,
                            cat_eigvals=self.hparams.cat_eigvals, cat_mult_eigvals=self.hparams.cat_mult_eigvals, no_extra_n=(not self.hparams.skip_noise_preprocess),
                            no_cond=self.hparams.no_cond, init_emb_channels=self.hparams.init_emb_channels, qm9=self.hparams.qm9,
                            data_channels_mult=self.hparams.ppgn_data_channels_mult)
        return generator

    def _get_SON_generator(self) -> nn.Module:
        SON_generator = SONGenerator(alpha=self.hparams.gen_leaky_ReLU_alpha, n_max=self.hparams.n_max,
                                    data_channels=self.hparams.hid_G,  gelu=self.hparams.gen_gelu, k_eigval=self.hparams.k_eigval, n_rot=self.hparams.n_rot, 
                                    dropout=self.hparams.G_dropout, max_pool=self.hparams.SON_max_pool, skip_connection=self.hparams.SON_skip_connection,
                                    share_weights=self.hparams.SON_share_weights, noise_latent_dim=self.hparams.noise_latent_dim,
                                    normalize_left=self.hparams.SON_normalize_left, no_extra_n=(not self.hparams.skip_noise_preprocess),
                                    small=self.hparams.SON_small,  init_bank_size=self.hparams.SON_init_bank_size, gumbel_temperature=self.hparams.SON_gumbel_temperature,
                                    kl_init_scale=self.hparams.SON_kl_init_scale, stiefel_sim_init=self.hparams.SON_stiefel_sim_init)
        return SON_generator

    def _get_lambda_generator(self) -> nn.Module:
        lambda_generator = LambdaGenerator(alpha=self.hparams.gen_leaky_ReLU_alpha, noise_latent_dim=self.hparams.noise_latent_dim,  data_channels=self.hparams.hid_G, gelu=self.hparams.gen_gelu,
                                            k_eigval=self.hparams.k_eigval, n_max=self.hparams.n_max,
                                            norm=self.hparams.lambda_norm, upsample=self.hparams.lambda_upsample,
                                            gating=self.hparams.lambda_gating, last_gating=self.hparams.lambda_last_gating, last_linear=self.hparams.lambda_last_linear,
                                            dropout=self.hparams.lambda_dropout)
        return lambda_generator

    def _get_discriminator(self) -> nn.Module:
        discriminator = PPGNDiscriminator(alpha=self.hparams.disc_leaky_ReLU_alpha, n_max=self.hparams.n_max, n_layers=self.hparams.n_D, data_channels=self.hparams.hid_D,
                                        use_spectral_norm=self.hparams.spectral_norm, normalization=self.hparams.disc_normalization, gelu=self.hparams.disc_gelu,
                                        k_eigval=self.hparams.k_eigval, dropout=self.hparams.D_dropout, cat_eigvals=self.hparams.cat_eigvals, cat_mult_eigvals=self.hparams.cat_mult_eigvals,
                                        partial_laplacian=self.hparams.D_partial_laplacian, no_cond=(self.hparams.no_cond or self.hparams.use_fixed_emb or self.hparams.mlp_gen),
                                        qm9=self.hparams.qm9, data_channels_mult=self.hparams.ppgn_data_channels_mult)
        return discriminator

    def _get_SON_discriminator(self) -> nn.Module:
        SON_discriminator = SONPointNetDiscriminator(alpha=self.hparams.disc_leaky_ReLU_alpha, n_max=self.hparams.n_max, data_channels=self.hparams.hid_D,
                                            use_spectral_norm=self.hparams.spectral_norm, gelu=self.hparams.disc_gelu,
                                            k_eigval=self.hparams.k_eigval, dropout=self.hparams.D_dropout,
                                            max_pool=False, skip_connection=self.hparams.SON_skip_connection,
                                            full_readout=self.hparams.SON_D_full_readout,
                                            n_rot=self.hparams.SON_D_n_rot, normalize_left=self.hparams.SON_normalize_left,
                                            small=self.hparams.SON_small)
        return SON_discriminator

    def _get_lambda_discriminator(self) -> nn.Module:
        lambda_discriminator = LambdaDiscriminator(alpha=self.hparams.disc_leaky_ReLU_alpha, data_channels=self.hparams.hid_D,
                                                    use_spectral_norm=self.hparams.spectral_norm, gelu=self.hparams.disc_gelu,
                                                    k_eigval=self.hparams.k_eigval, n_max=self.hparams.n_max,
                                                    norm=self.hparams.lambda_norm, gating=self.hparams.lambda_gating,
                                                    dropout=self.hparams.lambda_dropout)
        return lambda_discriminator

    def on_save_checkpoint(self, checkpoint):
        checkpoint['gen_ema_state_dict'] = self.gen_ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self.gen_ema.load_state_dict(checkpoint['gen_ema_state_dict'])
        self.gen_ema.to(self.device)

    def _eigval_temp(self, test_type: str = ''):
        if test_type == 'all_fake':
            return 0.0
        elif test_type == 'fake_adj' or test_type == 'fake_eigvec':
            return 1.0
        elif self.hparams.eigval_temp_decay:
            interval_len = self.hparams.n_eigval_warmup_epochs + self.hparams.decay_eigval_temp_over
            if self.current_epoch < self.hparams.n_eigval_warmup_epochs or (self.hparams.sharp_restart and (self.current_epoch % interval_len) < self.hparams.n_eigval_warmup_epochs):
                return self.hparams.max_eigval_temp
            elif (self.hparams.no_restart and self.current_epoch >= interval_len):
                return self.hparams.min_eigval_temp
            else:
                if self.hparams.sharp_restart:
                    current_step = (self.current_epoch % interval_len) - self.hparams.n_eigval_warmup_epochs
                else:
                    current_step = self.current_epoch - self.hparams.n_eigval_warmup_epochs
                temp = self.hparams.min_eigval_temp + (self.hparams.max_eigval_temp - self.hparams.min_eigval_temp) * 0.5 * (1 + math.cos(math.pi * (current_step) / self.hparams.decay_eigval_temp_over))
                return temp
        else:
            return 1.0

    def _mix_eigvals(self, real_eigval, fake_eigval, test_type: str = ''):
        real_eigval_mask = torch.bernoulli(torch.ones(real_eigval.size(0))*self._eigval_temp(test_type)).bool()
        fake_eigval = fake_eigval.clone()
        fake_eigval[real_eigval_mask] = real_eigval[:,:self.hparams.k_eigval][real_eigval_mask]
        return fake_eigval, real_eigval_mask

    def _eigvec_temp(self, test_type: str = ''):
        if test_type == 'all_fake' or test_type == 'fake_eigvec':
            return 0.0
        elif test_type == 'fake_adj':
            return 1.0
        elif self.hparams.eigvec_temp_decay:
            interval_len = self.hparams.n_eigvec_warmup_epochs + self.hparams.decay_eigvec_temp_over
            if self.current_epoch < self.hparams.n_eigvec_warmup_epochs or (self.hparams.sharp_restart and (self.current_epoch % interval_len) < self.hparams.n_eigvec_warmup_epochs):
                return self.hparams.max_eigvec_temp
            elif (self.hparams.no_restart and self.current_epoch >= interval_len):
                return self.hparams.min_eigvec_temp
            else:
                if self.hparams.sharp_restart:
                    current_step = (self.current_epoch % interval_len) - self.hparams.n_eigvec_warmup_epochs
                else:
                    current_step = self.current_epoch - self.hparams.n_eigvec_warmup_epochs
                temp = self.hparams.min_eigvec_temp + (self.hparams.max_eigvec_temp - self.hparams.min_eigvec_temp) * 0.5 * (1 + math.cos(math.pi * (current_step) / self.hparams.decay_eigvec_temp_over))
                return temp
        else:
            return 1.0

    def _mix_eigvecs(self, real_eigvec, fake_eigvec, real_eigval, mixed_eigval, real_eigval_mask, test_type: str = ''):
        real_eigvec_mask = torch.bernoulli(torch.ones(real_eigvec.size(0))*self._eigvec_temp(test_type)).bool()
        if self.hparams.temp_new:
            mixed_eigvec_eigval = mixed_eigval.clone()
            mixed_eigvec_eigval[real_eigvec_mask] = mixed_eigval[:,:self.hparams.k_eigval][real_eigvec_mask]
        else:
            real_eigvec_mask = torch.logical_and(real_eigvec_mask, real_eigval_mask) # Pass fake eigvecs if eigvals were fake
            mixed_eigvec_eigval = mixed_eigval
        fake_eigvec = fake_eigvec.clone()
        fake_eigvec[real_eigvec_mask] = real_eigvec[:,:,:self.hparams.k_eigval][real_eigvec_mask] 
        return fake_eigvec, real_eigvec_mask, mixed_eigvec_eigval

    def _SON_gumbel_temp(self):
        if self.hparams.SON_gumbel_temperature_decay:
            interval_len = self.hparams.SON_gumbel_temperature_warmup_epochs + self.hparams.decay_SON_gumbel_temp_over
            if self.current_epoch < self.hparams.SON_gumbel_temperature_warmup_epochs or (self.hparams.sharp_restart and (self.current_epoch % interval_len) < self.hparams.SON_gumbel_temperature_warmup_epochs):
                return self.hparams.SON_gumbel_temperature
            elif (self.hparams.no_restart and self.current_epoch >= interval_len):
                return self.hparams.min_SON_gumbel_temperature
            else:
                if self.hparams.sharp_restart:
                    current_step = (self.current_epoch % interval_len) - self.hparams.SON_gumbel_temperature_warmup_epochs
                else:
                    current_step = self.current_epoch - self.hparams.SON_gumbel_temperature_warmup_epochs
                temp = self.hparams.min_SON_gumbel_temperature + (self.hparams.SON_gumbel_temperature - self.hparams.min_SON_gumbel_temperature) * 0.5 * (1 + math.cos(math.pi * (current_step) / self.hparams.decay_SON_gumbel_temp_over))
                if temp <= 1e-3:
                    temp = 1e-3
                return temp
        else:
            return self.hparams.SON_gumbel_temperature

    def _get_noise(self, mask):
        batch_size = mask.size(0)
        n = torch.sum(mask, dim=-1, keepdim=True) / self.hparams.n_max
        noise = torch.randn([batch_size, self.hparams.noise_latent_dim], device=self.device)
        SON_node_noise = torch.randn([batch_size, mask.size(1), self.hparams.noise_latent_dim], device=mask.device)
        if self.hparams.mlp_gen:
            adj_node_noise = torch.randn([batch_size, self.hparams.noise_latent_dim], device=mask.device)
        else:
            adj_node_noise = torch.randn([batch_size, mask.size(1), self.hparams.noise_latent_dim], device=mask.device)
        
        SON_node_noise = SON_node_noise * mask[:,:,0].unsqueeze(-1)
        if not self.hparams.mlp_gen:
            adj_node_noise = adj_node_noise * mask[:,:,0].unsqueeze(-1)

        if self.hparams.normalize_noise:
            SON_node_noise = SON_node_noise / (torch.linalg.vector_norm(SON_node_noise, dim=-1, keepdim=True) + 1e-8)
            adj_node_noise = adj_node_noise / (torch.linalg.vector_norm(adj_node_noise, dim=-1, keepdim=True) + 1e-8)
            if self.hparams.derived_eigval_noise:
                noise = torch.mean(SON_node_noise, dim=1)
            noise = noise / (torch.linalg.vector_norm(noise, dim=-1, keepdim=True) + 1e-8)
        elif self.hparams.derived_eigval_noise:
            noise = torch.mean(SON_node_noise, dim=1)
        
        noise = self.mlp_noise(torch.cat([noise, n[:,0]], dim=-1))
        if not self.hparams.skip_noise_preprocess:
            SON_node_noise = self.mlp_SON_node_noise(torch.cat([SON_node_noise, n], dim=-1))
            if self.hparams.mlp_gen:
                adj_node_noise = self.mlp_adj_node_noise(torch.cat([adj_node_noise, n[:,0]], dim=-1))
            else:
                adj_node_noise = self.mlp_adj_node_noise(torch.cat([adj_node_noise, n], dim=-1))

        return noise, SON_node_noise, adj_node_noise

    def _get_fake(self, real_eigval: torch.Tensor, real_eigvec: torch.Tensor, mask: torch.Tensor, test_type: str = '', return_adj_noise: bool = False) -> torch.Tensor:
        noise, SON_node_noise, adj_node_noise = self._get_noise(mask)
        if self.hparams.adj_only:
            test_type = 'fake_adj'
            fake_eigval = real_eigval[:,:self.hparams.k_eigval]
            mixed_eigval = real_eigval[:,:self.hparams.k_eigval]
            fake_eigvec = real_eigvec[:,:,:self.hparams.k_eigval]
            if self.hparams.eigvec_sign_flip:
                fake_eigvec = deterministic_vector_sign_flip(fake_eigvec)
            mixed_eigvec = real_eigvec[:,:,:self.hparams.k_eigval]
            SON_aux_loss = 0.0
            mixed_eigvec_eigval = mixed_eigval
        elif self.hparams.lambda_only:
            fake_eigval = self.lambda_generator(noise, mask)
            if test_type != '':
                fake_eigval = fake_eigval[:,:self.hparams.k_eigval]
            return fake_eigval
        elif self.hparams.SON_only:
            fake_eigval = real_eigval[:,:self.hparams.k_eigval]
            mixed_eigval = real_eigval[:,:self.hparams.k_eigval]
            real_eigval_mask = torch.ones(real_eigval.size(0)).bool()
            fake_eigvec, SON_aux_loss = self.SON_generator(SON_node_noise, mixed_eigval, mask)
            if self.hparams.eigvec_sign_flip:
                fake_eigvec = deterministic_vector_sign_flip(fake_eigvec)
            return fake_eigvec, SON_aux_loss
        elif self.hparams.lambda_SON_only or (self.current_epoch < self.hparams.pretrain):
            fake_eigval = self.lambda_generator(noise, mask)
            if test_type != '':
                fake_eigval = fake_eigval[:,:self.hparams.k_eigval]
            mixed_eigval, real_eigval_mask = self._mix_eigvals(real_eigval, fake_eigval[:,:self.hparams.k_eigval], test_type=test_type)
            real_eigval_mask = torch.ones(real_eigval.size(0)).bool()
            fake_eigvec, SON_aux_loss = self.SON_generator(SON_node_noise, mixed_eigval, mask)
            if self.hparams.eigvec_sign_flip:
                fake_eigvec = deterministic_vector_sign_flip(fake_eigvec)
            return fake_eigvec, mixed_eigval, fake_eigval, SON_aux_loss
        elif self.hparams.adj_eigvec_only:
            fake_eigval = real_eigval[:,:self.hparams.k_eigval]
            mixed_eigval = real_eigval[:,:self.hparams.k_eigval]
            real_eigval_mask = torch.ones(real_eigval.size(0)).bool()
            fake_eigvec, SON_aux_loss = self.SON_generator(SON_node_noise, mixed_eigval, mask)
            if self.hparams.eigvec_sign_flip:
                fake_eigvec = deterministic_vector_sign_flip(fake_eigvec)
            mixed_eigvec, real_eigvec_mask, mixed_eigvec_eigval = self._mix_eigvecs(real_eigvec, fake_eigvec, real_eigval, mixed_eigval, real_eigval_mask, test_type=test_type)
        else: # Generate everything
            fake_eigval = self.lambda_generator(noise, mask)
            if test_type != '':
                fake_eigval = fake_eigval[:,:self.hparams.k_eigval]
            mixed_eigval, real_eigval_mask = self._mix_eigvals(real_eigval, fake_eigval[:,:self.hparams.k_eigval], test_type=test_type)
            fake_eigvec, SON_aux_loss = self.SON_generator(SON_node_noise, mixed_eigval, mask)
            if self.hparams.eigvec_sign_flip:
                fake_eigvec = deterministic_vector_sign_flip(fake_eigvec)
            mixed_eigvec, real_eigvec_mask, mixed_eigvec_eigval = self._mix_eigvecs(real_eigvec, fake_eigvec, real_eigval, mixed_eigval, real_eigval_mask, test_type=test_type)
        if self.hparams.qm9:
            fake_adj, fake_node_features, fake_edge_features = self.generator(adj_node_noise, mixed_eigvec_eigval, mixed_eigvec, mask)
            if return_adj_noise:
                return fake_adj, fake_node_features, fake_edge_features, mixed_eigvec, fake_eigvec, mixed_eigval, fake_eigval, mixed_eigvec_eigval, SON_aux_loss, adj_node_noise
            else:
                return fake_adj, fake_node_features, fake_edge_features, mixed_eigvec, fake_eigvec, mixed_eigval, fake_eigval, mixed_eigvec_eigval, SON_aux_loss
        else:
            fake_adj = self.generator(adj_node_noise, mixed_eigvec_eigval, mixed_eigvec, mask)
            if return_adj_noise:
                return fake_adj, None, None, mixed_eigvec, fake_eigvec, mixed_eigval, fake_eigval, mixed_eigvec_eigval, SON_aux_loss, adj_node_noise
            else:
                return fake_adj, None, None, mixed_eigvec, fake_eigvec, mixed_eigval, fake_eigval, mixed_eigvec_eigval, SON_aux_loss

    def _gradient_penalty(self, real_eigvec: torch.Tensor, fake_eigvec: torch.Tensor, real_eigval: torch.Tensor, fake_eigval: torch.Tensor, mask: torch.Tensor, real_adj: torch.Tensor,  fake_adj: torch.Tensor,
                            eigvec_pen: bool = False, eigval_pen: bool = False, do_backwards: bool = False, alpha: torch.Tensor = None,
                            real_node_features: torch.Tensor = None, real_edge_features: torch.Tensor = None, fake_node_features: torch.Tensor = None, fake_edge_features: torch.Tensor = None) -> torch.Tensor:
        # LP penalty (https://arxiv.org/pdf/1709.08894.pdf eq 8)
        gradient_penalty = 0.0

        if not self.hparams.gp_shared_alpha:
            alpha = torch.rand((real_eigval.size(0), 1), device=self.device)

        if eigval_pen:
            # Get random interpolation between real and fake data
            real_eigval_permuted = (alpha.expand(-1, real_eigval.size(1)) * real_eigval + ((1 - alpha.expand(-1, real_eigval.size(1))) * fake_eigval))
            inputs = [(True, real_eigval_permuted, None, None, None, None)]
        else:
            # Get random interpolation between real and fake data, but stay close to the real/fake data
            alpha_real = 1.0 - torch.rand((real_eigval.size(0), 1), device=self.device) / 4.0 # in (0.75, 1.0]
            alpha_fake = torch.rand((real_eigval.size(0), 1), device=self.device) / 4.0 # in [0, 0.25)
            real_eigval_permuted = (alpha_real.expand(-1, real_eigval.size(1)) * real_eigval + ((1 - alpha_real.expand(-1, real_eigval.size(1))) * fake_eigval))
            fake_eigval_permuted = (alpha_fake.expand(-1, real_eigval.size(1)) * real_eigval + ((1 - alpha_fake.expand(-1, real_eigval.size(1))) * fake_eigval))
            # Interpolate eigenvectors
            if not self.hparams.gp_shared_alpha:
                alpha_real = 1.0 - torch.rand((real_eigval.size(0), 1), device=self.device) / 4.0 # in (0.25, 1.0]
                alpha_fake = torch.rand((real_eigval.size(0), 1), device=self.device) / 4.0 # in [0, 0.25)
            real_eigvecs_permuted = interpolate_eigvecs(real_eigvec[:,:,:self.hparams.k_eigval], fake_eigvec[:,:,:self.hparams.k_eigval], mask=mask[:,:,0].unsqueeze(-1), alpha=alpha_real.view(-1,1,1).expand_as(fake_eigvec[:,:,:self.hparams.k_eigval]))
            fake_eigvecs_permuted = interpolate_eigvecs(real_eigvec[:,:,:self.hparams.k_eigval], fake_eigvec[:,:,:self.hparams.k_eigval], mask=mask[:,:,0].unsqueeze(-1), alpha=alpha_fake.view(-1,1,1).expand_as(fake_eigvec[:,:,:self.hparams.k_eigval]))
        
            if eigvec_pen:
                inputs = [(True, real_eigval_permuted, real_eigvecs_permuted, None, None, None), (False, fake_eigval_permuted, fake_eigvecs_permuted, None, None, None)]
                if self.hparams.gp_include_unpermuted:
                    inputs += [(True, real_eigval, real_eigvec, None, None, None), (False, fake_eigval, fake_eigvec, None, None, None)]
            else: # GP for everything
                if self.hparams.qm9:
                    real_adj_permuted, real_edge_features_permuted = rand_rewire(real_adj, mask=mask, fraction=self.hparams.gp_adj_rewire, noise=self.hparams.gp_adj_noise, edge_features=real_edge_features)
                    fake_adj_permuted, fake_edge_features_permuted = rand_rewire(fake_adj, mask=mask, fraction=self.hparams.gp_adj_rewire, noise=self.hparams.gp_adj_noise, edge_features=fake_edge_features)
                    fake_node_features_permuted = categorical_permute(fake_node_features, mask=mask[:,:,0].unsqueeze(-1), fraction=self.hparams.gp_adj_rewire, noise=self.hparams.gp_adj_noise)
                    real_node_features_permuted = categorical_permute(real_node_features, mask=mask[:,:,0].unsqueeze(-1), fraction=self.hparams.gp_adj_rewire, noise=self.hparams.gp_adj_noise)
                else:
                    real_adj_permuted = rand_rewire(real_adj, mask=mask, fraction=self.hparams.gp_adj_rewire, noise=self.hparams.gp_adj_noise)
                    fake_adj_permuted = rand_rewire(fake_adj, mask=mask, fraction=self.hparams.gp_adj_rewire, noise=self.hparams.gp_adj_noise)
                    fake_edge_features_permuted = fake_edge_features
                    real_edge_features_permuted = real_edge_features
                    fake_node_features_permuted = fake_node_features
                    real_node_features_permuted = real_node_features
                
                inputs = [(True, real_eigval_permuted, real_eigvecs_permuted, real_adj_permuted, real_node_features_permuted, real_edge_features_permuted), (False, fake_eigval_permuted, fake_eigvecs_permuted, fake_adj_permuted, fake_node_features_permuted, fake_edge_features_permuted)]
                if self.hparams.gp_include_unpermuted:
                    inputs += [(True, real_eigval, real_eigvec, real_adj, real_node_features, real_edge_features), (False, fake_eigval, fake_eigvec, fake_adj, fake_node_features,fake_edge_features)]
        
        for real, eigval, eigvec, adj, node_features, edge_features in inputs:
            eigval.requires_grad_(True)
            if eigval_pen:
                disc_interpolates = self.lambda_discriminator(eigval, mask)
            elif eigvec_pen:
                eigvec.requires_grad_(True)
                disc_interpolates = self.SON_discriminator(eigval, eigvec, mask)
            else: # Final disc GP
                eigvec.requires_grad_(True)
                adj.requires_grad_(True)
                if self.hparams.qm9:
                    node_features.requires_grad_(True)
                    edge_features.requires_grad_(True)
                disc_interpolates = self.discriminator(eigval, eigvec, mask, adj, node_features=node_features, edge_features=edge_features)

            grad_outputs = torch.ones(disc_interpolates.size(), device=self.device, requires_grad=False)

            if self.hparams.mlp_gen or self.hparams.no_cond or self.hparams.use_fixed_emb:
                if self.hparams.qm9:  
                    gradients = torch.autograd.grad(
                        outputs=disc_interpolates,
                        inputs=[node_features,edge_features,adj],
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )
                    penalty = self.hparams.gp_lambda * (torch.mean(torch.maximum(((gradients[0] * mask[:,:,0].unsqueeze(-1)).norm(2, dim=[1,2]) - 1), torch.zeros(1, device=gradients[1].device)) ** 2) +
                                                        torch.mean(torch.maximum(((gradients[1] * mask.unsqueeze(-1)).norm(2, dim=[1,2,3]) - 1), torch.zeros(1, device=gradients[2].device)) ** 2) + 
                                                        torch.mean(torch.maximum(((gradients[2] * mask).norm(2, dim=[1,2]) - 1), torch.zeros(1, device=gradients[2].device)) ** 2)) / (len(inputs) * 3.0)
                else:
                    gradients = torch.autograd.grad(
                        outputs=disc_interpolates,
                        inputs=adj,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    penalty = self.hparams.gp_lambda * torch.mean(torch.maximum(((gradients * mask).norm(2, dim=[1,2]) - 1), torch.zeros(1, device=gradients.device)) ** 2) / (len(inputs))
            else:
                if eigval_pen:
                    gradients = torch.autograd.grad(
                        outputs=disc_interpolates,
                        inputs=eigval,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    gradients = gradients.reshape(gradients.size(0), -1)
                    penalty = self.hparams.gp_lambda * torch.mean(torch.maximum((gradients.norm(2, dim=1) - 1), torch.zeros(1, device=gradients.device)) ** 2) / (len(inputs))
                elif eigvec_pen:
                    gradients = torch.autograd.grad(
                        outputs=disc_interpolates,
                        inputs=[eigval, eigvec],
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )
                    penalty = self.hparams.gp_lambda * (torch.mean(torch.maximum(((gradients[0]).norm(2, dim=1) - 1), torch.zeros(1, device=gradients[0].device)) ** 2) + 
                                                        torch.mean(torch.maximum(((gradients[1] * mask[:,:,0].unsqueeze(-1)).norm(2, dim=[1,2]) - 1), torch.zeros(1, device=gradients[1].device)) ** 2) ) / (len(inputs) * 2.0)
                else: # Final disc GP
                    if self.hparams.qm9:  
                        gradients = torch.autograd.grad(
                            outputs=disc_interpolates,
                            inputs=[node_features,edge_features,eigval,eigvec,adj],
                            grad_outputs=grad_outputs,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )
                        penalty = self.hparams.gp_lambda * (torch.mean(torch.maximum(((gradients[0] * mask[:,:,0].unsqueeze(-1)).norm(2, dim=[1,2]) - 1), torch.zeros(1, device=gradients[1].device)) ** 2) +
                                                            torch.mean(torch.maximum(((gradients[1] * mask.unsqueeze(-1)).norm(2, dim=[1,2,3]) - 1), torch.zeros(1, device=gradients[2].device)) ** 2) + 
                                                            torch.mean(torch.maximum(((gradients[2]).norm(2, dim=1) - 1), torch.zeros(1, device=gradients[0].device)) ** 2) + 
                                                            torch.mean(torch.maximum(((gradients[3] * mask[:,:,0].unsqueeze(-1)).norm(2, dim=[1,2]) - 1), torch.zeros(1, device=gradients[1].device)) ** 2) +
                                                            torch.mean(torch.maximum(((gradients[4] * mask).norm(2, dim=[1,2]) - 1), torch.zeros(1, device=gradients[2].device)) ** 2)) / (len(inputs) * 5.0)
                    else:
                        gradients = torch.autograd.grad(
                            outputs=disc_interpolates,
                            inputs=[eigval,eigvec,adj],
                            grad_outputs=grad_outputs,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )
                        penalty = self.hparams.gp_lambda * (torch.mean(torch.maximum(((gradients[0]).norm(2, dim=1) - 1), torch.zeros(1, device=gradients[0].device)) ** 2) + 
                                                            torch.mean(torch.maximum(((gradients[1] * mask[:,:,0].unsqueeze(-1)).norm(2, dim=[1,2]) - 1), torch.zeros(1, device=gradients[1].device)) ** 2) +
                                                            torch.mean(torch.maximum(((gradients[2] * mask).norm(2, dim=[1,2]) - 1), torch.zeros(1, device=gradients[2].device)) ** 2)) / (len(inputs) * 3.0)

            if torch.isnan(penalty).any():
                print('penalty', real, eigvec_pen, eigval_pen)
                print(disc_interpolates)
                print(gradients[0])
                print(eigval)
                print(torch.sum(mask[:,0], dim=-1))
                raise ValueError
            if do_backwards:
                self.manual_backward(penalty)
                penalty = penalty.item()
            gradient_penalty = gradient_penalty + penalty

        return gradient_penalty

    def _disc_step(self, real_eigval: torch.Tensor, real_eigvec: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor, real_edge_features: torch.Tensor, real_node_features: torch.Tensor) -> torch.Tensor:
        # Sort eigenvecs in canonical order:
        real_eigvec, eigvec_indices = sort_eigvecs(real_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)
        if self.hparams.qm9:
            real_node_features = real_node_features.gather(1, eigvec_indices.unsqueeze(-1).expand_as(real_node_features))
            adj, real_edge_features = reorder_adj(adj, eigvec_indices, real_edge_features)
        else:
            adj = reorder_adj(adj, eigvec_indices)

        if self.hparams.gp_shared_alpha:
            # Use the same interpolation parameter for eigenvalues and eigenvectors
            gp_alpha = torch.rand((real_eigval.size(0), 1), device=self.device)
        else:
            gp_alpha = None

        # Add a bit of noise to eigvals/eigvecs to not overfit
        if self.hparams.noisy_disc:
            noisy_real_eigval = eigval_noise(real_eigval, variance=self.hparams.eigval_noise)
            noisy_real_eigvec = rand_rot(real_eigvec, variance=self.hparams.eigvec_noise, right_noise=self.hparams.eigvec_right_noise)
        else:
            noisy_real_eigval = real_eigval
            noisy_real_eigvec = real_eigvec

        noisy_real_eigvec, noisy_real_eigvec_indices = sort_eigvecs(noisy_real_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)
        if self.hparams.qm9:
            noisy_real_node_features = real_node_features.gather(1, noisy_real_eigvec_indices.unsqueeze(-1).expand_as(real_node_features))
            noisy_adj, noisy_real_edge_features = reorder_adj(adj, noisy_real_eigvec_indices, real_edge_features)
        else:
            noisy_real_node_features = real_node_features
            noisy_real_edge_features = real_edge_features
            noisy_adj = reorder_adj(adj, noisy_real_eigvec_indices)

        real_eigval = real_eigval[:, :self.hparams.k_eigval]
        noisy_real_eigval = noisy_real_eigval[:, :self.hparams.k_eigval]

        # Condition on noise free eigenvalues and eigenvectors
        cond_real_eigval = real_eigval
        cond_real_eigvec = real_eigvec
        
        if self.hparams.lambda_only:
            self.lambda_generator.eval()
            with torch.no_grad():
                fake_eigval = self._get_fake(cond_real_eigval, cond_real_eigvec, mask)
            self.lambda_generator.train()
            opt = self.optimizers(use_pl_optimizer=True)[0]
            opt.zero_grad(set_to_none=True)

            lambda_fake_pred = self.lambda_discriminator(fake_eigval, mask)
            lambda_real_pred = self.lambda_discriminator(real_eigval, mask)

            lambda_disc_loss = lambda_fake_pred.mean() - lambda_real_pred.mean()
            if self.hparams.wgan_eps > 0:
                lambda_disc_loss += (lambda_real_pred.mean() ** 2) * self.hparams.wgan_eps   
            
            disc_loss = self.hparams.lambda_disc * lambda_disc_loss
            lambda_penalty = self._gradient_penalty(None, None, fake_eigval, real_eigval, mask, None, None, eigval_pen=True, do_backwards=False, alpha=gp_alpha)
            disc_loss = lambda_disc_loss + lambda_penalty
            
            self.manual_backward(disc_loss)

            opt.step()

            self.log("loss/lambda_disc_loss", lambda_disc_loss)
            self.log("loss/lambda_penalty", lambda_penalty)
        elif self.hparams.SON_only:
            self.SON_generator.eval()
            with torch.no_grad():
                fake_eigvec, SON_aux_loss = self._get_fake(cond_real_eigval, cond_real_eigvec, mask)
                fake_eigvec, _ = sort_eigvecs(fake_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)
            self.SON_generator.train()
            opt = self.optimizers(use_pl_optimizer=True)[0]
            opt.zero_grad(set_to_none=True)

            if self.hparams.noisy_disc:
                noisy_cond_real_eigval = eigval_noise(cond_real_eigval, variance=self.hparams.eigval_noise)
                noisy_fake_eigvec = rand_rot(fake_eigvec, variance=self.hparams.eigvec_noise, right_noise=self.hparams.eigvec_right_noise)
            else:
                noisy_cond_real_eigval = cond_real_eigval
                noisy_fake_eigvec = fake_eigvec
            
            noisy_fake_eigvec, _ = sort_eigvecs(noisy_fake_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)
            
            son_fake_pred = self.SON_discriminator(noisy_cond_real_eigval, fake_eigvec, mask)
            son_real_pred = self.SON_discriminator(noisy_real_eigval, real_eigvec, mask)

            SON_disc_loss = son_fake_pred.mean() - son_real_pred.mean()  
            if self.hparams.wgan_eps > 0:
                SON_disc_loss += (son_real_pred.mean() ** 2) * self.hparams.wgan_eps         
            disc_loss = self.hparams.SON_disc * SON_disc_loss

            SON_penalty = self._gradient_penalty(real_eigvec, fake_eigvec, real_eigval, cond_real_eigval, mask, None, None, eigvec_pen=True, do_backwards=False, alpha=gp_alpha)
            disc_loss = SON_disc_loss + SON_penalty
            self.manual_backward(disc_loss)

            opt.step()

            self.log("loss/SON_disc_loss", SON_disc_loss)
            self.log("loss/SON_penalty", SON_penalty)
        elif self.hparams.lambda_SON_only or (self.current_epoch < self.hparams.pretrain):
            # Train with real
            real_pred = self.discriminator(noisy_real_eigval, noisy_real_eigvec, mask, noisy_adj)
            self.log("loss/disc_real", real_pred.mean())

            # Train with fake
            self.generator.eval() # Disable dropout.
            self.SON_generator.eval()
            self.lambda_generator.eval()
            with torch.no_grad():
                fake_eigvec, mixed_eigval, fake_eigval, SON_aux_loss = self._get_fake(cond_real_eigval, cond_real_eigvec, mask)
                fake_eigvec, _ = sort_eigvecs(fake_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)

            opt = self.optimizers(use_pl_optimizer=True)[0]
            opt.zero_grad(set_to_none=True)

            if self.hparams.noisy_disc:
                noisy_mixed_eigval = eigval_noise(mixed_eigval, variance=self.hparams.eigval_noise)
                noisy_fake_eigvec = rand_rot(fake_eigvec, variance=self.hparams.eigvec_noise, right_noise=self.hparams.eigvec_right_noise)
            else:
                noisy_mixed_eigval = mixed_eigval
                noisy_fake_eigvec = fake_eigvec
            
            noisy_fake_eigvec, _ = sort_eigvecs(noisy_fake_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)

            if self.hparams.SON_disc > 0 and not self.hparams.adj_only:
                son_fake_pred = self.SON_discriminator(noisy_mixed_eigval, fake_eigvec, mask)
                son_real_pred = self.SON_discriminator(noisy_real_eigval, real_eigvec, mask)

                SON_disc_loss = son_fake_pred.mean() - son_real_pred.mean()  
                if self.hparams.wgan_eps > 0:
                    SON_disc_loss += (son_real_pred.mean() ** 2) * self.hparams.wgan_eps          
                
            else:
                SON_disc_loss = 0.0

            if self.hparams.lambda_disc > 0 and not self.hparams.adj_eigvec_only and not self.hparams.adj_only and not self.hparams.SON_only:
                lambda_fake_pred = self.lambda_discriminator(fake_eigval, mask)
                lambda_real_pred = self.lambda_discriminator(real_eigval, mask)

                lambda_disc_loss = lambda_fake_pred.mean() - lambda_real_pred.mean()
                if self.hparams.wgan_eps > 0:
                    lambda_disc_loss += (lambda_real_pred.mean() ** 2) * self.hparams.wgan_eps   
                
            else:
                lambda_disc_loss = 0.0

            disc_loss = self.hparams.SON_disc * SON_disc_loss + self.hparams.lambda_disc * lambda_disc_loss

            if self.hparams.SON_disc > 0 and not self.hparams.adj_only:
                SON_penalty = self._gradient_penalty(real_eigvec, fake_eigvec, real_eigval, mixed_eigval, mask, None, None, eigvec_pen=True, do_backwards=False, alpha=gp_alpha)
                disc_loss = disc_loss + SON_penalty
            if self.hparams.lambda_disc > 0 and not self.hparams.adj_eigvec_only and not self.hparams.adj_only and not self.hparams.SON_only:
                lambda_penalty = self._gradient_penalty(None, None, fake_eigval, real_eigval, mask, None, None, eigval_pen=True, do_backwards=False, alpha=gp_alpha) 
                disc_loss = disc_loss + lambda_penalty
            self.manual_backward(disc_loss)
            
            self.log("loss/disc_loss", disc_loss)
            self.log("loss/SON_disc_loss", SON_disc_loss)
            self.log("loss/lambda_disc_loss", lambda_disc_loss)
            if self.hparams.SON_disc > 0 and not self.hparams.adj_only:
                self.log("loss/lambda_penalty", lambda_penalty)
            if self.hparams.lambda_disc > 0 and not self.hparams.adj_eigvec_only and not self.hparams.adj_only and not self.hparams.SON_only:
                self.log("loss/SON_penalty", SON_penalty)
            opt.step()
        else:
            # Train with real
            real_pred = self.discriminator(noisy_real_eigval, noisy_real_eigvec, mask, noisy_adj, node_features=noisy_real_node_features, edge_features=noisy_real_edge_features)

            # Train with fake
            self.generator.eval() # Disable dropout.
            self.SON_generator.eval()
            self.lambda_generator.eval()
            with torch.no_grad():
                fake_adj, fake_node_features, fake_edge_features, mixed_eigvec, fake_eigvec, mixed_eigval, fake_eigval, mixed_eigvec_eigval, SON_aux_loss = self._get_fake(cond_real_eigval, cond_real_eigvec, mask)
                fake_eigvec, eigvec_indices = sort_eigvecs(fake_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)
                mixed_eigvec, mixed_eigvec_indices = sort_eigvecs(mixed_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)
                if self.hparams.qm9:
                    fake_node_features = fake_node_features.gather(1, mixed_eigvec_indices.unsqueeze(-1).expand_as(fake_node_features))
                    fake_adj, fake_edge_features = reorder_adj(fake_adj, mixed_eigvec_indices, fake_edge_features)
                else:
                    fake_adj = reorder_adj(fake_adj, mixed_eigvec_indices)

            self.generator.train()
            self.SON_generator.train()
            self.lambda_generator.train()

            if self.hparams.noisy_disc:
                noisy_mixed_eigval = eigval_noise(mixed_eigval, variance=self.hparams.eigval_noise)
                noisy_mixed_eigvec_eigval = eigval_noise(mixed_eigvec_eigval, variance=self.hparams.eigval_noise)
                noisy_fake_eigvec = rand_rot(fake_eigvec, variance=self.hparams.eigvec_noise, right_noise=self.hparams.eigvec_right_noise)
                noisy_mixed_eigvec = rand_rot(mixed_eigvec, variance=self.hparams.eigvec_noise, right_noise=self.hparams.eigvec_right_noise)
            else:
                noisy_mixed_eigval = mixed_eigval
                noisy_fake_eigvec = fake_eigvec
                noisy_mixed_eigvec_eigval = mixed_eigvec_eigval
                noisy_mixed_eigvec = mixed_eigvec
            
            noisy_mixed_eigvec, noisy_mixed_eigvec_indices = sort_eigvecs(noisy_mixed_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)
            if self.hparams.qm9:
                noisy_fake_node_features = fake_node_features.gather(1, noisy_mixed_eigvec_indices.unsqueeze(-1).expand_as(fake_node_features))
                noisy_fake_adj, noisy_fake_edge_features = reorder_adj(fake_adj, noisy_mixed_eigvec_indices, fake_edge_features)
            else:
                noisy_fake_node_features = fake_node_features
                noisy_fake_edge_features = fake_edge_features
                noisy_fake_adj = reorder_adj(fake_adj, noisy_mixed_eigvec_indices)
            
            fake_pred = self.discriminator(noisy_mixed_eigvec_eigval, noisy_mixed_eigvec, mask, noisy_fake_adj, node_features=noisy_fake_node_features, edge_features=noisy_fake_edge_features)

            adj_loss = fake_pred.mean() - real_pred.mean()
            if self.hparams.wgan_eps > 0:
                adj_loss += (real_pred.mean() ** 2) * self.hparams.wgan_eps

            disc_loss = adj_loss

            if self.hparams.SON_disc > 0 and not self.hparams.adj_only:
                son_fake_pred = self.SON_discriminator(noisy_mixed_eigval, fake_eigvec, mask)
                son_real_pred = self.SON_discriminator(noisy_real_eigval, real_eigvec, mask)

                SON_disc_loss = son_fake_pred.mean() - son_real_pred.mean()  
                if self.hparams.wgan_eps > 0:
                    SON_disc_loss += (son_real_pred.mean() ** 2) * self.hparams.wgan_eps     
                
                SON_disc_loss = self.hparams.SON_disc * SON_disc_loss
            else:
                SON_disc_loss = 0.0

            if self.hparams.lambda_disc > 0 and not self.hparams.adj_eigvec_only and not self.hparams.adj_only and not self.hparams.SON_only:
                lambda_fake_pred = self.lambda_discriminator(fake_eigval, mask)
                lambda_real_pred = self.lambda_discriminator(real_eigval, mask)

                lambda_disc_loss = lambda_fake_pred.mean() - lambda_real_pred.mean()
                if self.hparams.wgan_eps > 0:
                    lambda_disc_loss += (lambda_real_pred.mean() ** 2) * self.hparams.wgan_eps
                lambda_disc_loss = self.hparams.lambda_disc * lambda_disc_loss
            else:
                lambda_disc_loss = 0.0

            nan_vals = False
            if torch.isnan(real_pred).any():
                print('real_pred')
                nan_vals = True
            if torch.isnan(fake_adj).any():
                print('fake_adj')
                nan_vals = True
            if torch.isnan(fake_pred).any():
                print('fake_pred')
                nan_vals = True
            if torch.isnan(adj_loss).any():
                print('adj_loss')
                nan_vals = True
            if nan_vals:
                print('NaNs found')
            
            disc_loss_no_pen = adj_loss + SON_disc_loss + lambda_disc_loss
            disc_loss = disc_loss_no_pen

            opt = self.optimizers(use_pl_optimizer=True)[0]
            opt.zero_grad(set_to_none=True)

            if self.hparams.SON_disc > 0 and not self.hparams.adj_only:
                SON_penalty = self._gradient_penalty(real_eigvec, fake_eigvec, real_eigval, mixed_eigval, mask, None, None, eigvec_pen=True, do_backwards=False, alpha=gp_alpha)         
                disc_loss = disc_loss + SON_penalty
            else:
                SON_penalty = 0.0
            if self.hparams.lambda_disc > 0 and not self.hparams.adj_eigvec_only and not self.hparams.adj_only and not self.hparams.SON_only:
                lambda_penalty = self._gradient_penalty(None, None, fake_eigval, real_eigval, mask, None, None, eigval_pen=True, do_backwards=False, alpha=gp_alpha)
                disc_loss = disc_loss + lambda_penalty
            else:
                lambda_penalty = 0.0

            if self.hparams.gp_do_backwards: # Save a lot of memory, by doing backwards after each forward pass through the PPGN
                self.manual_backward(disc_loss)
                adj_penalty = self._gradient_penalty(real_eigvec, mixed_eigvec, real_eigval, mixed_eigvec_eigval, mask, adj, fake_adj, do_backwards=True, alpha=gp_alpha, real_node_features=real_node_features, real_edge_features=real_edge_features, fake_node_features=fake_node_features, fake_edge_features=fake_edge_features)
                disc_loss = disc_loss + adj_penalty
            else: 
                adj_penalty = self._gradient_penalty(real_eigvec, mixed_eigvec, real_eigval, mixed_eigvec_eigval, mask, adj, fake_adj, do_backwards=False, alpha=gp_alpha, real_node_features=real_node_features, real_edge_features=real_edge_features, fake_node_features=fake_node_features, fake_edge_features=fake_edge_features)
                disc_loss = disc_loss + adj_penalty
                self.manual_backward(disc_loss)
        
            opt.step()
            self.log("loss/disc_real", real_pred.mean())
            self.log("loss/disc_fake", fake_pred.mean())
            self.log("loss/base_disc_loss", adj_loss)  
            self.log("loss/SON_disc_loss", SON_disc_loss)
            self.log("loss/lambda_disc_loss", lambda_disc_loss)
            self.log("loss/disc_loss", disc_loss_no_pen)
            self.log("loss/lambda_penalty", lambda_penalty)
            self.log("loss/SON_penalty", SON_penalty)
            self.log("loss/penalty", adj_penalty)

        disc_loss = disc_loss.item()
        self.log("loss/disc", disc_loss)

    def _gen_step(self, real_eigval: torch.Tensor, real_eigvec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Disable dropouts
        self.discriminator.eval()
        self.SON_discriminator.eval()
        self.lambda_discriminator.eval()

        real_eigval = real_eigval[:, :self.hparams.k_eigval]
        if self.hparams.noisy_gen:
            cond_real_eigval = eigval_noise(real_eigval, variance=self.hparams.eigval_noise)
            cond_real_eigvec = rand_rot(real_eigvec, variance=self.hparams.eigvec_noise, right_noise=self.hparams.eigvec_right_noise)
        else:
            cond_real_eigval = real_eigval
            cond_real_eigvec = real_eigvec
        
        # Train with fake
        if self.hparams.lambda_only:
            fake_eigval = self._get_fake(cond_real_eigval, cond_real_eigvec, mask)

            lambda_fake_pred = self.lambda_discriminator(fake_eigval, mask)
            fake_pred = self.hparams.lambda_disc * lambda_fake_pred
            
            gen_loss = -fake_pred.mean()

            opt = self.optimizers(use_pl_optimizer=True)[1]
            opt.zero_grad(set_to_none=True)
            self.manual_backward(gen_loss)
            if self.hparams.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.gen_params, self.hparams.clip_grad_norm)
            opt.step()
            self.log("loss/lambda_gen_aux", -lambda_fake_pred.mean())

        elif self.hparams.SON_only:
            fake_eigvec, SON_aux_loss = self._get_fake(cond_real_eigval, cond_real_eigvec, mask)

            son_fake_pred = self.SON_discriminator(cond_real_eigval, fake_eigvec, mask)
            
            fake_pred = self.hparams.SON_disc * son_fake_pred
            
            gen_loss = -fake_pred.mean() + SON_aux_loss

            opt = self.optimizers(use_pl_optimizer=True)[1]
            opt.zero_grad(set_to_none=True)
            self.manual_backward(gen_loss)
            if self.hparams.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.gen_params, self.hparams.clip_grad_norm)
            opt.step()
            self.log("loss/SON_gen_aux", -son_fake_pred.mean())

        elif self.hparams.lambda_SON_only or (self.current_epoch < self.hparams.pretrain):
            fake_eigvec, mixed_eigval, fake_eigval, SON_aux_loss = self._get_fake(cond_real_eigval, cond_real_eigvec, mask)
            
            son_fake_pred = self.SON_discriminator(mixed_eigval, fake_eigvec, mask)
            
            fake_pred = self.hparams.SON_disc * son_fake_pred

            lambda_fake_pred = self.lambda_discriminator(fake_eigval, mask)
            
            fake_pred = fake_pred + self.hparams.lambda_disc * lambda_fake_pred
            
            gen_loss = -fake_pred.mean() + SON_aux_loss

            opt = self.optimizers(use_pl_optimizer=True)[1]
            opt.zero_grad(set_to_none=True)
            self.manual_backward(gen_loss)
            if self.hparams.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.gen_params, self.hparams.clip_grad_norm)
            opt.step()
            self.log("loss/SON_gen_aux", -son_fake_pred.mean())
            self.log("loss/lambda_gen_aux", -lambda_fake_pred.mean())

        else:
            fake_adj, fake_node_features, fake_edge_features, mixed_eigvec, fake_eigvec, mixed_eigval, fake_eigval, mixed_eigvec_eigval, SON_aux_loss = self._get_fake(cond_real_eigval, cond_real_eigvec, mask)
            fake_pred_adj = self.discriminator(mixed_eigvec_eigval, mixed_eigvec, mask, fake_adj, node_features=fake_node_features, edge_features=fake_edge_features)
            fake_pred = fake_pred_adj
            
            if self.hparams.SON_disc > 0 and not self.hparams.adj_only:
                son_fake_pred = self.SON_discriminator(mixed_eigval, fake_eigvec, mask)
                fake_pred = fake_pred + self.hparams.SON_disc * son_fake_pred
            

            if self.hparams.lambda_disc > 0 and not self.hparams.adj_eigvec_only and not self.hparams.adj_only and not self.hparams.SON_only:
                lambda_fake_pred = self.lambda_discriminator(fake_eigval, mask)
                fake_pred = fake_pred + self.hparams.lambda_disc * lambda_fake_pred
            
            gen_loss = -fake_pred.mean() + SON_aux_loss
            
            gen_loss = gen_loss

            opt = self.optimizers(use_pl_optimizer=True)[1]
            opt.zero_grad(set_to_none=True)
            self.manual_backward(gen_loss)
            if self.hparams.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.gen_params, self.hparams.clip_grad_norm)
            opt.step()
            self.log("loss/gen_main", -fake_pred_adj.mean()) 
            if self.hparams.SON_disc > 0 and not self.hparams.adj_only:
                self.log("loss/SON_gen_aux", -son_fake_pred.mean())
            else:
                self.log("loss/SON_gen_aux", 0.0)
            if self.hparams.lambda_disc > 0 and not self.hparams.adj_eigvec_only and not self.hparams.adj_only and not self.hparams.SON_only:
                self.log("loss/lambda_gen_aux", -lambda_fake_pred.mean())
            else:
                self.log("loss/lambda_gen_aux", 0.0)

        self.discriminator.train()
        self.SON_discriminator.train()
        self.lambda_discriminator.train()

        if self.gen_ema.shadow_params[0].device != self.device:
            self.gen_ema.to(self.device)
        self.gen_ema.update(self.gen_params)

        gen_loss = gen_loss.item()
        self.log("loss/gen", gen_loss) 

    def forward(self, node_noise: torch.Tensor, eigval: torch.Tensor, eigvec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.generator(node_noise, eigval, eigvec, mask)

    def training_step(self, batch, batch_idx):

        torch.cuda.reset_peak_memory_stats(0)
        batch_max_n = max(batch['n_nodes'].max(), self.hparams.k_eigval)
        # real_eigval = batch["eigval"][:, :self.hparams.k_eigval]
        real_eigval = batch["eigval"]
        real_eigvec = batch["eigvec"][:, :batch_max_n, :self.hparams.k_eigval]
        mask = batch["mask"][:, :batch_max_n, :batch_max_n]
        adj = batch["adj"][:, :batch_max_n, :batch_max_n]
        if self.hparams.qm9:
            real_edge_features = batch["edge_features"][:, :batch_max_n, :batch_max_n]
            real_node_features = batch["node_features"][:, :batch_max_n]
        else:
            real_edge_features = None
            real_node_features = None
        
        self.SON_generator.gumbel_temperature = self._SON_gumbel_temp()

        # Log sampling temperature
        self.log("epoch/eigvec_temp", self._eigvec_temp())
        self.log("epoch/eigval_temp", self._eigval_temp())
        if not self.hparams.temp_new:
            self.log("epoch/effective_eigvec_temp", self._eigval_temp() * self._eigvec_temp())
        self.log("epoch/SON_Gumbel_temp", self.SON_generator.gumbel_temperature)

        # Train discriminator
        self._disc_step(real_eigval, real_eigvec, mask, adj, real_edge_features, real_node_features)

        # Train generator
        self._gen_step(real_eigval, real_eigvec, mask)

        self.log("epoch/max_allocated_mem", float(round(torch.cuda.max_memory_allocated()/1024.0**2)))
        self.log("epoch/max_cached_mem", float(round(torch.cuda.max_memory_reserved()/1024.0**2)))

        sch_D, sch_G = self.lr_schedulers()
        self.log("epoch/lr_D", sch_D.get_last_lr()[0])
        self.log("epoch/lr_G", sch_G.get_last_lr()[0])

        if self.trainer.is_last_batch and self.hparams.SON_init_bank_size > 0:
            # A weird way to track how many elements from the SON init sample bank where used in the eopch
            self.log("loss/bank_sample_hist", (self.SON_generator.bank_sample_hist > 0).float().sum(), on_epoch=True, on_step=False)
            self.SON_generator.bank_sample_hist = torch.zeros_like(self.SON_generator.bank_sample_hist)

        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % self.hparams.lr_decay_every == 0 and (self.trainer.current_epoch + 1) >= self.hparams.lr_decay_warmup:
            sch_D.step()
            sch_G.step()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out_dict = {}
        # Run validation on EMA params and current step params
        if self.hparams.ema > 0.0:
            with self.gen_ema.average_parameters():
                out_dict = out_dict | self.validation_function(batch, batch_idx, dataloader_idx=dataloader_idx, out_dict=out_dict, suffix='_ema')
        out_dict = out_dict | self.validation_function(batch, batch_idx, dataloader_idx=dataloader_idx, out_dict=out_dict, suffix='')

        val_loss = -self.current_epoch #decreases so that lightining saves checkpoint every val_step, might be unecessary with new versions
        out_dict = {**out_dict, f'val_loss': val_loss}
        return out_dict

    def validation_function(self, batch, batch_idx, dataloader_idx=0, out_dict={}, suffix=''):
        batch_max_n = max(batch['n_nodes'].max(), self.hparams.k_eigval)
        n_nodes = batch['n_nodes']
        real_eigval = batch["eigval"][:, :self.hparams.k_eigval]
        real_eigvec = batch["eigvec"][:, :batch_max_n, :self.hparams.k_eigval]
        mask = batch["mask"][:, :batch_max_n, :batch_max_n]
        adj_true = batch["adj"][:, :batch_max_n, :batch_max_n]

        cmap = cm.get_cmap('rainbow', self.hparams.k_eigval) 

        if dataloader_idx == 1:
            cond_data_sufix = '_train_cond'
        else:
            cond_data_sufix = ''
        with torch.no_grad():
            if self.hparams.lambda_only:
                for test_type in ['all_fake']:
                    true_eigvals = []
                    fake_eigvals = []

                    fake_eigval = self._get_fake(real_eigval, real_eigvec, mask, test_type=test_type)

                    for i in range(real_eigval.size(0)):
                        true_eigvals.append(real_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvals.append(fake_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())

                    if batch_idx==0:
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                        ax.plot(real_eigval[0].cpu().detach().numpy())
                        ax.plot(fake_eigval[0].cpu().detach().numpy())
                        self.logger.experiment.add_figure(f'generated_eigvals_{test_type}{cond_data_sufix}{suffix}', fig, self.current_epoch)
                        plt.close(fig)

                        plt.close('all')
                    out_dict = {**out_dict, f'{test_type}_true_eigvals{suffix}': true_eigvals, f'{test_type}_fake_eigvals{suffix}': fake_eigvals}
            elif self.hparams.SON_only:
                for test_type in ['fake_eigvec']:
                    true_eigvals = []
                    fake_eigvals = []
                    true_eigvecs = []
                    fake_eigvecs = []

                    fake_eigvec, SON_aux_loss = self._get_fake(real_eigval, real_eigvec, mask, test_type=test_type)
                    # Use canonical eigenvector sort for visualization
                    fake_eigvec, _ = sort_eigvecs(fake_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)
                    real_eigvec, _ = sort_eigvecs(real_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)

                    for i in range(real_eigval.size(0)):
                        true_eigvals.append(real_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvals.append(real_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        true_eigvecs.append(real_eigvec[i, :n_nodes[i], :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvecs.append(fake_eigvec[i, :n_nodes[i], :self.hparams.k_eigval].cpu().detach().numpy())

                    SON_loss = fake_eigvec @ real_eigvec.transpose(-2,-1)
                    SON_loss = SON_loss - torch.eye(SON_loss.size(1), SON_loss.size(2), device=SON_loss.device).unsqueeze(0)
                    SON_loss = torch.mean(SON_loss ** 2)

                    if batch_idx == 0:
                        for i in range(len(real_eigvec)):
                            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                            ax.plot(real_eigvec[i,:,0].cpu().detach().numpy())
                            ax.plot(fake_eigvec[i,:,0].cpu().detach().numpy())
                            self.logger.experiment.add_figure(f'generated_eigvec_1_{test_type}{cond_data_sufix}{suffix}/{i}', fig, self.current_epoch)
                            plt.close(fig)

                            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                            ax.plot(real_eigvec[i,:,1].cpu().detach().numpy())
                            ax.plot(fake_eigvec[i,:,1].cpu().detach().numpy())
                            self.logger.experiment.add_figure(f'generated_eigvec_2_{test_type}{cond_data_sufix}{suffix}/{i}', fig, self.current_epoch)
                            plt.close(fig)
                        
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                        if self.hparams.ignore_first_eigv:
                            ax.scatter(real_eigvec[0,:n_nodes[0],0].cpu().detach().numpy(), real_eigvec[0,:n_nodes[0],1].cpu().detach().numpy(), c=real_eigvec[0,:n_nodes[0],1].cpu().detach().numpy())
                        else:
                            ax.scatter(real_eigvec[0,:n_nodes[0],1].cpu().detach().numpy(), real_eigvec[0,:n_nodes[0],2].cpu().detach().numpy(), c=real_eigvec[0,:n_nodes[0],1].cpu().detach().numpy())
                        self.logger.experiment.add_figure(f'real_eigvec_scatter_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)
                        plt.close(fig)

                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                        if self.hparams.ignore_first_eigv:
                            ax.scatter(fake_eigvec[0,:n_nodes[0],0].cpu().detach().numpy(), fake_eigvec[0,:n_nodes[0],1].cpu().detach().numpy(), c=fake_eigvec[0,:n_nodes[0],1].cpu().detach().numpy())
                        else:
                            ax.scatter(fake_eigvec[0,:n_nodes[0],1].cpu().detach().numpy(), fake_eigvec[0,:n_nodes[0],2].cpu().detach().numpy(), c=fake_eigvec[0,:n_nodes[0],1].cpu().detach().numpy())
                        self.logger.experiment.add_figure(f'generated_eigvec_scatter_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)
                        plt.close(fig)

                    out_dict = {**out_dict, f'{test_type}_SON_loss{suffix}': SON_loss, f'{test_type}_true_eigvals{suffix}': true_eigvals, f'{test_type}_fake_eigvals{suffix}': fake_eigvals,
                                f'{test_type}_true_eigvecs{suffix}': true_eigvecs, f'{test_type}_fake_eigvecs{suffix}': fake_eigvecs}
            elif self.hparams.lambda_SON_only or (self.current_epoch < self.hparams.pretrain):
                for test_type in ['all_fake', 'fake_eigvec']:
                    true_eigvals = []
                    fake_eigvals = []
                    true_eigvecs = []
                    fake_eigvecs = []

                    fake_eigvec, mixed_eigval, fake_eigval, SON_aux_loss = self._get_fake(real_eigval, real_eigvec, mask, test_type=test_type)
                    fake_eigvec, _ = sort_eigvecs(fake_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)
                    real_eigvec, _ = sort_eigvecs(real_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)

                    for i in range(real_eigval.size(0)):
                        true_eigvals.append(real_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvals.append(mixed_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        true_eigvecs.append(real_eigvec[i, :n_nodes[i], :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvecs.append(fake_eigvec[i, :n_nodes[i], :self.hparams.k_eigval].cpu().detach().numpy())

                    SON_loss = fake_eigvec @ real_eigvec.transpose(-2,-1)
                    SON_loss = SON_loss - torch.eye(SON_loss.size(1), SON_loss.size(2), device=SON_loss.device).unsqueeze(0)
                    SON_loss = torch.mean(SON_loss ** 2)

                    if batch_idx==0:
                        # Plot true and fake eigvectors
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                        ax.plot(real_eigvec[0,:,0].cpu().detach().numpy())
                        ax.plot(fake_eigvec[0,:,0].cpu().detach().numpy())
                        self.logger.experiment.add_figure(f'generated_eigvec_1_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)
                        plt.close(fig)

                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                        ax.plot(real_eigvec[0,:,1].cpu().detach().numpy())
                        ax.plot(fake_eigvec[0,:,1].cpu().detach().numpy())
                        self.logger.experiment.add_figure(f'generated_eigvec_2_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)
                        plt.close(fig)

                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                        if self.hparams.ignore_first_eigv:
                            ax.scatter(real_eigvec[0,:n_nodes[0],0].cpu().detach().numpy(), real_eigvec[0,:n_nodes[0],1].cpu().detach().numpy(), c=real_eigvec[0,:n_nodes[0],1].cpu().detach().numpy())
                        else:
                            ax.scatter(real_eigvec[0,:n_nodes[0],1].cpu().detach().numpy(), real_eigvec[0,:n_nodes[0],2].cpu().detach().numpy(), c=real_eigvec[0,:n_nodes[0],1].cpu().detach().numpy())
                        self.logger.experiment.add_figure(f'real_eigvec_scatter_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)
                        plt.close(fig)

                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                        if self.hparams.ignore_first_eigv:
                            ax.scatter(fake_eigvec[0,:n_nodes[0],0].cpu().detach().numpy(), fake_eigvec[0,:n_nodes[0],1].cpu().detach().numpy(), c=fake_eigvec[0,:n_nodes[0],1].cpu().detach().numpy())
                        else:
                            ax.scatter(fake_eigvec[0,:n_nodes[0],1].cpu().detach().numpy(), fake_eigvec[0,:n_nodes[0],2].cpu().detach().numpy(), c=fake_eigvec[0,:n_nodes[0],1].cpu().detach().numpy())
                        self.logger.experiment.add_figure(f'generated_eigvec_scatter_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)
                        plt.close(fig)

                        # Plot true and fake eigvals
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                        ax.plot(real_eigval[0].cpu().detach().numpy())
                        ax.plot(mixed_eigval[0].cpu().detach().numpy())
                        self.logger.experiment.add_figure(f'generated_eigvals_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)
                        plt.close(fig)
                        
                        # Vizualize some other random initializations for the same conditioning
                        fake_eigvec_i_batch, mixed_eigval_i_batch, fake_eigval_i_batch, SON_aux_loss  = self._get_fake(real_eigval[0].unsqueeze(0).expand(2, -1), real_eigvec[0].unsqueeze(0).expand(2, -1, -1), mask[0].unsqueeze(0).expand(2, -1, -1), test_type=test_type)
                        fake_eigvec_i_batch, _ = sort_eigvecs(fake_eigvec_i_batch, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)

                        for i in range(1,3):
                            fake_eigvec_i, mixed_eigval_i, fake_eigval_i = fake_eigvec_i_batch[i-1], mixed_eigval_i_batch[i-1], fake_eigval_i_batch[i-1]

                            # Plot true and fake eigvectors
                            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                            ax.plot(real_eigvec[0,:,0].cpu().detach().numpy())
                            ax.plot(fake_eigvec_i[:,0].cpu().detach().numpy())
                            self.logger.experiment.add_figure(f'generated_eigvec_1_{test_type}{cond_data_sufix}{suffix}/{i}', fig, self.current_epoch)
                            plt.close(fig)

                            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                            ax.plot(real_eigvec[0,:,1].cpu().detach().numpy())
                            ax.plot(fake_eigvec_i[:,1].cpu().detach().numpy())
                            self.logger.experiment.add_figure(f'generated_eigvec_2_{test_type}{cond_data_sufix}{suffix}/{i}', fig, self.current_epoch)
                            plt.close(fig)
                            
                            if not (self.hparams.mlp_gen or self.hparams.no_cond or self.hparams.use_fixed_emb):
                                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                                if self.hparams.ignore_first_eigv:
                                    ax.scatter(fake_eigvec_i[:n_nodes[0],0].cpu().detach().numpy(), fake_eigvec_i[:n_nodes[0],1].cpu().detach().numpy(), c=fake_eigvec_i[:n_nodes[0],1].cpu().detach().numpy())
                                else:
                                    ax.scatter(fake_eigvec_i[:n_nodes[0],1].cpu().detach().numpy(), fake_eigvec_i[:n_nodes[0],2].cpu().detach().numpy(), c=fake_eigvec_i[:n_nodes[0],1].cpu().detach().numpy())
                                self.logger.experiment.add_figure(f'generated_eigvec_scatter_{test_type}{cond_data_sufix}{suffix}/{i}', fig, self.current_epoch)
                                plt.close(fig)

                            # Plot true and fake eigvals
                            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                            ax.plot(real_eigval[0].cpu().detach().numpy())
                            ax.plot(mixed_eigval_i.cpu().detach().numpy())
                            self.logger.experiment.add_figure(f'generated_eigvals_{test_type}{cond_data_sufix}{suffix}/{i}', fig, self.current_epoch)
                            plt.close(fig)

                    out_dict = {**out_dict, f'{test_type}_SON_loss{suffix}': SON_loss, f'{test_type}_true_eigvals{suffix}': true_eigvals, f'{test_type}_fake_eigvals{suffix}': fake_eigvals,
                                f'{test_type}_true_eigvecs{suffix}': true_eigvecs, f'{test_type}_fake_eigvecs{suffix}': fake_eigvecs}

            else:
                if self.hparams.adj_only:
                    test_types = ['fake_adj']
                elif self.hparams.adj_eigvec_only:
                    test_types = ['fake_eigvec', 'fake_adj']
                else:
                    test_types = ['all_fake', 'fake_eigvec', 'fake_adj']
                for test_type in test_types:
                    true_graphs = []
                    fake_graphs = []
                    true_eigvals = []
                    fake_eigvals = []
                    true_eigvecs = []
                    fake_eigvecs = []
                    if self.hparams.qm9:
                        fake_node_features = []
                        fake_edge_features = []
                        fake_adj = []

                    adj, node_features, edge_features, mixed_eigvec, fake_eigvec, mixed_eigval, fake_eigval, mixed_eigvec_eigval, SON_aux_loss = self._get_fake(real_eigval, real_eigvec, mask, test_type=test_type)

                    # Discretize adjacency:
                    adj[adj < 0] = 0.0 
                    adj = zero_diag(torch.round(adj))
                    adj[adj > 1] = 1.0 

                    if self.hparams.ignore_first_eigv:
                        eigv_offset = 1
                    else:
                        eigv_offset = 0

                    if self.hparams.qm9:
                        # Discretize node and edge classes:
                        node_features = torch.argmax(node_features, dim=-1)
                        # Only keep edge features for existing edges
                        edge_features = (torch.argmax(edge_features, dim=-1) + 1) * adj # +1 for conversion to weighted nx matrix

                    for i in range(adj.size(0)):
                        true_eigvals.append(real_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvals.append(mixed_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        true_eigvecs.append(real_eigvec[i, :n_nodes[i], :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvecs.append(fake_eigvec[i, :n_nodes[i], :self.hparams.k_eigval].cpu().detach().numpy())
                        if self.hparams.qm9:
                            #  Add edge class as edege "weight"
                            nx_graph = nx.from_numpy_array(edge_features[i, :n_nodes[i], :n_nodes[i]].cpu().detach().numpy())
                            #  Add node features  to nx graph
                            for j in range(n_nodes[i].item()):
                                nx_graph.nodes[j]["class"] = node_features[i, j].item()
                            fake_graphs.append(nx_graph)
                            fake_node_features.append(node_features[i, :n_nodes[i]].cpu().detach().long())
                            fake_edge_features.append((edge_features[i, :n_nodes[i], :n_nodes[i]] - 1).cpu().detach().long()) #Strat edge class indexing from 0 for BasicMolecularMetrics
                            fake_adj.append(adj[i, :n_nodes[i], :n_nodes[i]].cpu().detach().long())
                            out_dict = {**out_dict, f'{test_type}_fake_node_features{suffix}': fake_node_features, f'{test_type}_fake_edge_features{suffix}': fake_edge_features, f'{test_type}_fake_adj{suffix}': fake_adj}
                        else:
                            true_graphs.append(nx.from_numpy_array(adj_true[i, :n_nodes[i], :n_nodes[i]].cpu().detach().numpy()))
                            fake_graphs.append(nx.from_numpy_array(adj[i, :n_nodes[i], :n_nodes[i]].cpu().detach().numpy()))
                    if batch_idx==0:
                        # Use canonical eigenvector sort for visualization
                        fake_eigvec, _ = sort_eigvecs(fake_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)
                        mixed_eigvec, mixed_eigvec_indices = sort_eigvecs(mixed_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)
                        adj = reorder_adj(adj, mixed_eigvec_indices)
                        real_eigvec, real_eigvec_indices = sort_eigvecs(real_eigvec, mask[:,:,0], sign_flip=self.hparams.eigvec_sign_flip)
                        adj_true = reorder_adj(adj_true, real_eigvec_indices)

                        plt.figure()
                        plt.imshow(adj[0].cpu().detach().numpy(), interpolation='none')
                        plt.colorbar()
                        self.logger.experiment.add_figure(f'intermediate_fake_adjacency_matrices_{test_type}{cond_data_sufix}{suffix}/fake_0', plt.gcf(), self.current_epoch)
                        plt.close(plt.gcf())

                        fig = plt.figure(figsize=(8, 8))
                        G = nx.convert_matrix.from_numpy_array(adj[0, :n_nodes[0], :n_nodes[0]].cpu().detach().numpy())
                        pos = nx.spring_layout(G, iterations=50)
                        kmeans = KMeans(n_clusters=min(self.hparams.k_eigval, n_nodes[0].item()), n_init=100).fit(fake_eigvec.cpu().detach().numpy()[0,:n_nodes[0],:self.hparams.k_eigval])
                        y_pred = kmeans.labels_
                        nx.draw(G, pos, font_size=5, node_size=12, with_labels=True, node_color=cmap(y_pred))
                        self.logger.experiment.add_figure(f'generated_graph_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)#self.global_step
                        plt.close(fig)

                        deg = adj[0].sum(dim=-1)
                        D = torch.diag_embed(deg.abs()**(-0.5))
                        L = torch.eye(adj.size(1), adj.size(2), out=torch.zeros_like(adj[0])) - D @ adj[0] @ D
                        L[L.isnan()] = 0
                        del D
                        L = L[:n_nodes[0], :n_nodes[0]]

                        try:
                            rec_eigval, rec_eigvec = torch.linalg.eigh(L)
                            rec_eigvec = sort_eigvecs(rec_eigvec.unsqueeze(0), mask[0,:,0].unsqueeze(0), sign_flip=self.hparams.eigvec_sign_flip)[0].squeeze(0)

                            # Plot true and fake eigvectors
                            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                            ax.plot(real_eigvec[0,:,0].cpu().detach().numpy())
                            ax.plot(rec_eigvec[:,0+eigv_offset].cpu().detach().numpy())
                            ax.plot(mixed_eigvec[0,:,0].cpu().detach().numpy())
                            self.logger.experiment.add_figure(f'generated_eigvec_1_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)
                            plt.close(fig)

                            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                            ax.plot(real_eigvec[0,:,1].cpu().detach().numpy())
                            ax.plot(rec_eigvec[:,1+eigv_offset].cpu().detach().numpy())
                            ax.plot(mixed_eigvec[0,:,1].cpu().detach().numpy())
                            self.logger.experiment.add_figure(f'generated_eigvec_2_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)
                            plt.close(fig)
                            
                            if not (self.hparams.mlp_gen or self.hparams.no_cond or self.hparams.use_fixed_emb):
                                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                                if self.hparams.ignore_first_eigv:
                                    ax.scatter(fake_eigvec[0,:n_nodes[0],0].cpu().detach().numpy(), fake_eigvec[0,:n_nodes[0],1].cpu().detach().numpy(), c=fake_eigvec[0,:n_nodes[0],1].cpu().detach().numpy())
                                else:
                                    ax.scatter(fake_eigvec[0,:n_nodes[0],1].cpu().detach().numpy(), fake_eigvec[0,:n_nodes[0],2].cpu().detach().numpy(), c=fake_eigvec[0,:n_nodes[0],1].cpu().detach().numpy())
                                self.logger.experiment.add_figure(f'generated_eigvec_scatter_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)
                                plt.close(fig)

                            # Plot true and fake eigvals
                            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                            ax.plot(real_eigval[0].cpu().detach().numpy())
                            ax.plot(rec_eigval[eigv_offset:self.hparams.k_eigval+eigv_offset].cpu().detach().numpy())
                            ax.plot(mixed_eigvec_eigval[0].cpu().detach().numpy())
                            self.logger.experiment.add_figure(f'generated_eigvals_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)
                            plt.close(fig)
                        except RuntimeError:
                            pass
                        
                        # Vizualize some other random initializations for the same conditioning
                        adj_i_batch, node_features_i_batch, edge_features_i_batch, mixed_eigvec_i_batch, fake_eigvec_i_batch, mixed_eigval_i_batch, fake_eigval_i_batch, mixed_eigvec_eigval_i_batch, SON_aux_loss  = self._get_fake(real_eigval[0].unsqueeze(0).expand(2, -1), real_eigvec[0].unsqueeze(0).expand(2, -1, -1), mask[0].unsqueeze(0).expand(2, -1, -1), test_type=test_type)
                        # Use canonical eigenvector sort for visualization
                        fake_eigvec_i_batch, _ = sort_eigvecs(fake_eigvec_i_batch, mask[0,:,0].unsqueeze(0).expand(2, -1), sign_flip=self.hparams.eigvec_sign_flip)
                        mixed_eigvec_i_batch, mixed_eigvec_i_batch_indices = sort_eigvecs(mixed_eigvec_i_batch, mask[0,:,0].unsqueeze(0).expand(2, -1), sign_flip=self.hparams.eigvec_sign_flip)
                        adj_i_batch = reorder_adj(adj_i_batch, mixed_eigvec_i_batch_indices)

                        for i in range(1,3):
                            adj_i, mixed_eigvec_i, fake_eigvec_i, mixed_eigval_i, fake_eigval_i, mixed_eigvec_eigval_i = adj_i_batch[i-1], mixed_eigvec_i_batch[i-1], fake_eigvec_i_batch[i-1], mixed_eigval_i_batch[i-1], fake_eigval_i_batch[i-1], mixed_eigvec_eigval_i_batch[i-1]
                            
                            plt.figure()
                            plt.imshow(adj_i.cpu().detach().numpy(), interpolation='none')
                            plt.colorbar()
                            self.logger.experiment.add_figure(f'intermediate_fake_adjacency_matrices_{test_type}{cond_data_sufix}{suffix}/fake_{i}', plt.gcf(), self.current_epoch)
                            plt.close(plt.gcf())

                            # Discretize adjacency:
                            adj_i[adj_i < 0] = 0.0 
                            adj_i = torch.round(adj_i).fill_diagonal_(0, wrap=False)
                            adj_i[adj_i > 1] = 1.0 

                            fig = plt.figure(figsize=(8, 8))
                            G = nx.convert_matrix.from_numpy_array(adj_i[:n_nodes[0], :n_nodes[0]].cpu().detach().numpy())
                            pos = nx.spring_layout(G, iterations=50)
                            kmeans = KMeans(n_clusters=min(self.hparams.k_eigval, n_nodes[0].item()), n_init=100).fit(mixed_eigvec_i.cpu().detach().numpy()[:n_nodes[0],:self.hparams.k_eigval])
                            y_pred = kmeans.labels_
                            nx.draw(G, pos, font_size=5, node_size=12, with_labels=True, node_color=cmap(y_pred))
                            self.logger.experiment.add_figure(f'generated_graph_{test_type}{cond_data_sufix}{suffix}/{i}', fig, self.current_epoch)#self.global_step
                            plt.close(fig)

                            deg_i = adj_i.sum(dim=-1)
                            D = torch.diag_embed(deg_i.abs()**(-0.5))
                            L = torch.eye(adj_i.size(0), adj_i.size(1), out=torch.zeros_like(adj_i)) - D @ adj_i @ D
                            L[L.isnan()] = 0
                            del D
                            L = L[:n_nodes[0], :n_nodes[0]]

                            try:
                                rec_eigval_i, rec_eigvec_i = torch.linalg.eigh(L)
                                rec_eigvec_i = sort_eigvecs(rec_eigvec_i.unsqueeze(0), mask[0,:,0].unsqueeze(0), sign_flip=self.hparams.eigvec_sign_flip)[0].squeeze(0)

                                # Plot true and fake eigvectors
                                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                                ax.plot(real_eigvec[0,:,0].cpu().detach().numpy())
                                ax.plot(rec_eigvec_i[:,0+eigv_offset].cpu().detach().numpy())
                                ax.plot(mixed_eigvec_i[:,0].cpu().detach().numpy())
                                self.logger.experiment.add_figure(f'generated_eigvec_1_{test_type}{cond_data_sufix}{suffix}/{i}', fig, self.current_epoch)
                                plt.close(fig)

                                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                                ax.plot(real_eigvec[0,:,1].cpu().detach().numpy())
                                ax.plot(rec_eigvec_i[:,1+eigv_offset].cpu().detach().numpy())
                                ax.plot(mixed_eigvec_i[:,1].cpu().detach().numpy())
                                self.logger.experiment.add_figure(f'generated_eigvec_2_{test_type}{cond_data_sufix}{suffix}/{i}', fig, self.current_epoch)
                                plt.close(fig)

                                if not (self.hparams.mlp_gen or self.hparams.no_cond or self.hparams.use_fixed_emb):
                                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                                    if self.hparams.ignore_first_eigv:
                                        ax.scatter(mixed_eigvec_i[:n_nodes[0],0].cpu().detach().numpy(), mixed_eigvec_i[:n_nodes[0],1].cpu().detach().numpy(), c=cmap(y_pred))
                                    else:
                                        ax.scatter(mixed_eigvec_i[:n_nodes[0],1].cpu().detach().numpy(), mixed_eigvec_i[:n_nodes[0],2].cpu().detach().numpy(), c=cmap(y_pred))
                                    self.logger.experiment.add_figure(f'generated_eigvec_scatter_{test_type}{cond_data_sufix}{suffix}/{i}', fig, self.current_epoch)
                                    plt.close(fig)

                                # Plot true and fake eigvals
                                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                                ax.plot(real_eigval[0].cpu().detach().numpy())
                                ax.plot(rec_eigval_i[eigv_offset:self.hparams.k_eigval+eigv_offset].cpu().detach().numpy())
                                ax.plot(mixed_eigvec_eigval_i.cpu().detach().numpy())
                                self.logger.experiment.add_figure(f'generated_eigvals_{test_type}{cond_data_sufix}{suffix}/{i}', fig, self.current_epoch)
                                plt.close(fig)
                            except RuntimeError:
                                pass

                        if test_type == 'all_fake' or self.hparams.adj_only:
                            plt.figure()
                            plt.imshow(adj_true[0].cpu().detach().numpy(), interpolation='none')
                            plt.colorbar()
                            self.logger.experiment.add_figure(f'real_adjacency{cond_data_sufix}{suffix}', plt.gcf(), self.current_epoch)
                            plt.close(plt.gcf())

                            fig = plt.figure(figsize=(8, 8))
                            G = nx.convert_matrix.from_numpy_array(adj_true[0, :n_nodes[0], :n_nodes[0]].cpu().detach().numpy())
                            pos = nx.spring_layout(G, iterations=50)
                            kmeans = KMeans(n_clusters=min(self.hparams.k_eigval, n_nodes[0].item()), n_init=100).fit(real_eigvec.cpu().detach().numpy()[0,:n_nodes[0],:self.hparams.k_eigval])
                            y_pred = kmeans.labels_
                            nx.draw(G, pos, font_size=5, node_size=12, with_labels=True, node_color=cmap(y_pred))
                            self.logger.experiment.add_figure(f'true_graph{cond_data_sufix}{suffix}', fig, self.current_epoch)#self.global_step
                            plt.close(fig)
                            
                            if not (self.hparams.mlp_gen or self.hparams.no_cond or self.hparams.use_fixed_emb):
                                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                                if self.hparams.ignore_first_eigv:
                                    ax.scatter(real_eigvec[0,:n_nodes[0],0].cpu().detach().numpy(), real_eigvec[0,:n_nodes[0],1].cpu().detach().numpy(), c=cmap(y_pred))
                                else:
                                    ax.scatter(real_eigvec[0,:n_nodes[0],1].cpu().detach().numpy(), real_eigvec[0,:n_nodes[0],2].cpu().detach().numpy(), c=cmap(y_pred))
                                self.logger.experiment.add_figure(f'real_eigvec_scatter_{test_type}{cond_data_sufix}{suffix}/0', fig, self.current_epoch)
                                plt.close(fig)

                        plt.close('all')

                    # Compute L2 distance between input eigvalues and eigvalues of discretized generated adjacency matrix
                    deg = adj.sum(dim=-1)
                    D = torch.diag_embed(deg.abs()**(-0.5))
                    L = torch.eye(adj.size(1), adj.size(2), out=torch.zeros_like(adj)) - (D @ adj @ D)
                    L[L.isnan()] = 0
                    del D
                    spectral_fail = torch.tensor(0.0, device=self.device)
                    spectral_loss = -torch.ones(L.size(0), device=L.device)
                    normalized_spectral_loss = -torch.ones(L.size(0), device=L.device)
                    for i in range(L.size(0)):
                        try:
                            rec_eigval = torch.linalg.eigvalsh(L[i, :n_nodes[i], :n_nodes[i]])[eigv_offset:]
                            eigval_count = min(n_nodes[i].item(), self.hparams.k_eigval)
                            spectral_loss[i] = torch.mean((mixed_eigvec_eigval[i, :eigval_count][mixed_eigvec_eigval[i, :eigval_count] > 0] - rec_eigval[:eigval_count][mixed_eigvec_eigval[i, :eigval_count] > 0])**2)
                            normalized_spectral_loss[i] = torch.mean(torch.abs((mixed_eigvec_eigval[i, :eigval_count] - rec_eigval[:eigval_count])[mixed_eigvec_eigval[i, :eigval_count] > 0] / mixed_eigvec_eigval[i, :eigval_count][mixed_eigvec_eigval[i, :eigval_count] > 0]))
                        except RuntimeError:
                            print('Decomp error:')
                            print(n_nodes[i])
                            print(mixed_eigval[i])
                            print(adj[i, :n_nodes[i], :n_nodes[i]])
                            print(L[i, :n_nodes[i], :n_nodes[i]])
                            print(adj_true[i, :n_nodes[i], :n_nodes[i]], flush=True)
                            spectral_fail += 1.0
                    spectral_fail = spectral_fail / L.size(0)
                    spectral_loss = spectral_loss.mean()
                    normalized_spectral_loss = normalized_spectral_loss.mean()
                    # u^T @ L @ u - lambda - take only the number of eigvectors used to condition the model
                    spectral_match_loss = torch.tensor(0.0, device=self.device)
                    normalized_spectral_match_loss = torch.tensor(0.0, device=self.device)
                    for i in range(0, self.hparams.k_eigval):
                        spectral_match = (mixed_eigvec[:, :, i].view(mixed_eigvec.size(0),1,-1) @ L @ mixed_eigvec[:, :, i].view(mixed_eigvec.size(0),-1,1) - mixed_eigvec_eigval[:,i].view(mixed_eigvec_eigval.size(0),1,1))              
                        spectral_match_loss += torch.mean(spectral_match**2) / self.hparams.k_eigval
                        normalized_spectral_match_loss += ( torch.mean(spectral_match.abs()[mixed_eigvec_eigval[:,i]>0] / mixed_eigvec_eigval[:,i][mixed_eigvec_eigval[:,i]>0].view(mixed_eigvec_eigval[:,i][mixed_eigvec_eigval[:,i]>0].size(0),1,1)) ) / self.hparams.k_eigval
                    
                    out_dict = {**out_dict, f'{test_type}_true_graphs{suffix}': true_graphs, f'{test_type}_fake_graphs{suffix}': fake_graphs, f'{test_type}_true_eigvals{suffix}': true_eigvals, f'{test_type}_fake_eigvals{suffix}': fake_eigvals,
                                        f'{test_type}_spectral_loss{suffix}': spectral_loss, f'{test_type}_normalized_spectral_loss{suffix}': normalized_spectral_loss,
                                        f'{test_type}_spectral_match_loss{suffix}': spectral_match_loss, f'{test_type}_normalized_spectral_match_loss{suffix}': normalized_spectral_match_loss, f'{test_type}_spectral_fail{suffix}': spectral_fail,
                                        f'{test_type}_true_eigvecs{suffix}': true_eigvecs, f'{test_type}_fake_eigvecs{suffix}': fake_eigvecs}
        
        return out_dict

    def validation_epoch_end(self, outputs):
        if self.hparams.SON_init_bank_size:
            self.log("validation_loss/bank_sample_hist", (self.SON_generator.bank_sample_hist > 0).float().sum(), on_epoch=True, on_step=False)
            self.SON_generator.bank_sample_hist = torch.zeros_like(self.SON_generator.bank_sample_hist)

        recon_losses = []
        normalized_recon_losses = []
        if type(outputs[0]) == list:
            cond_data = ['', '_train_cond']
            single = False
        else:
            cond_data = ['']
            single = True
        if self.hparams.ema > 0.0:
            model_types = ['', '_ema']
        else:
            model_types = ['']
        for model_type in model_types:
            for dataloader_idx, cond_data_sufix in enumerate(cond_data):
                if not single:
                    outputs_i = outputs[dataloader_idx]
                else:
                    outputs_i = outputs
                if self.hparams.lambda_only:
                    for test_type in ['all_fake']:
                        true_eigvals = [g for x in outputs_i for g in x[f'{test_type}_true_eigvals{model_type}']]
                        fake_eigvals = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvals{model_type}']] 
                        mmd_eigval = eigval_stats(true_eigvals, fake_eigvals, max_eig=self.trainer.datamodule.max_k_eigval)
                        emd_eigval = eigval_stats(true_eigvals, fake_eigvals, max_eig=self.trainer.datamodule.max_k_eigval, compute_emd=True)
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/mmd_eigval', mmd_eigval)
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/emd_eigval', emd_eigval)
                        
                        mean_MMD_ratio = (mmd_eigval / (self.trainer.datamodule.train_mmd_spectral+ 1e-6))
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/mean_MMD_ratio', mean_MMD_ratio)
                        self.log(f'validation_loss/mean_recon_loss', 0)
                        self.log(f'validation_loss/mean_normalized_recon_loss', 0)
                        recon_losses.append(0)
                        normalized_recon_losses.append(0)
                elif self.hparams.SON_only:
                    test_types = ['fake_eigvec']
                    for test_type in test_types:
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/SON_loss', torch.stack([x[f'{test_type}_SON_loss{model_type}'] for x in outputs_i]).mean())

                        true_eigvals = [g for x in outputs_i for g in x[f'{test_type}_true_eigvals{model_type}']]
                        fake_eigvals = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvals{model_type}']] 

                        true_eigvecs = [g for x in outputs_i for g in x[f'{test_type}_true_eigvecs{model_type}']]
                        fake_eigvecs = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvecs{model_type}']] 
                        if self.hparams.compute_emd:
                            measure_types =  ['mmd', 'emd']
                        else:
                            measure_types =  ['mmd']
                        for measure_type in measure_types:
                            mmd_wavelet_eigvec = spectral_filter_stats(true_eigvecs, true_eigvals, fake_eigvecs, fake_eigvals, compute_emd=(measure_type=='emd'))
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_spectral_filter', mmd_wavelet_eigvec)

                    self.log(f'validation_loss/mean_MMD_ratio', 0)
                    self.log(f'validation_loss/mean_recon_loss', 0)
                    self.log(f'validation_loss/mean_normalized_recon_loss', 0)
                    recon_losses.append(0)
                    normalized_recon_losses.append(0)
                elif self.hparams.lambda_SON_only or (self.current_epoch < self.hparams.pretrain):
                    test_types = ['all_fake', 'fake_eigvec']
                    for test_type in test_types:
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/SON_loss', torch.stack([x[f'{test_type}_SON_loss{model_type}'] for x in outputs_i]).mean())

                        true_eigvals = [g for x in outputs_i for g in x[f'{test_type}_true_eigvals{model_type}']]
                        fake_eigvals = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvals{model_type}']] 

                        true_eigvecs = [g for x in outputs_i for g in x[f'{test_type}_true_eigvecs{model_type}']]
                        fake_eigvecs = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvecs{model_type}']] 

                        mmd_eigval = eigval_stats(true_eigvals, fake_eigvals, max_eig=self.trainer.datamodule.max_k_eigval)
                        emd_eigval = eigval_stats(true_eigvals, fake_eigvals, max_eig=self.trainer.datamodule.max_k_eigval, compute_emd=True)
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/mmd_eigval', mmd_eigval)
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/emd_eigval', emd_eigval)
                        
                        mean_MMD_ratio = ((mmd_eigval + 1e-8) / (self.trainer.datamodule.train_mmd_spectral + 1e-8))
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/mean_MMD_ratio', mean_MMD_ratio)
                        if self.hparams.compute_emd:
                            measure_types =  ['mmd', 'emd']
                        else:
                            measure_types =  ['mmd']
                        for measure_type in measure_types:
                            mmd_wavelet_eigvec = spectral_filter_stats(true_eigvecs, true_eigvals, fake_eigvecs, fake_eigvals, compute_emd=(measure_type=='emd'))
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_spectral_filter', mmd_wavelet_eigvec)

                    self.log(f'validation_loss/mean_recon_loss', 0)
                    self.log(f'validation_loss/mean_normalized_recon_loss', 0)
                    recon_losses.append(0)
                    normalized_recon_losses.append(0)
                else:
                    if self.hparams.adj_only:
                        test_types = ['fake_adj']
                    elif self.hparams.adj_eigvec_only:
                        test_types = ['fake_eigvec', 'fake_adj']
                    else:
                        test_types = ['all_fake', 'fake_eigvec', 'fake_adj']
                    for test_type in test_types:
                        true_graphs = [g for x in outputs_i for g in x[f'{test_type}_true_graphs{model_type}']]
                        fake_graphs = [g for x in outputs_i for g in x[f'{test_type}_fake_graphs{model_type}']] 

                        if self.hparams.qm9:
                            fake_node_features = [g for x in outputs_i for g in x[f'{test_type}_fake_node_features{model_type}']] 
                            fake_edge_features = [g for x in outputs_i for g in x[f'{test_type}_fake_edge_features{model_type}']] 
                            fake_adj = [g for x in outputs_i for g in x[f'{test_type}_fake_adj{model_type}']] 
                            molecule_metrics, _ =  self.trainer.datamodule.molecular_metrics.evaluate(list(zip(fake_node_features, fake_adj, fake_edge_features)))
                            valid, unique, novel = molecule_metrics
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/frac_valid', valid)
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/frac_unique', unique)
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/frac_novel', novel)
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/valid_and_unique', valid*unique) # unique is returned as fraction of valid
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/valid_and_unique_and_novel', valid*unique*novel) # unique is returned as fraction of valid
                        else:
                            # Get MMD measures
                            true_eigvals = [g for x in outputs_i for g in x[f'{test_type}_true_eigvals{model_type}']]
                            fake_eigvals = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvals{model_type}']] 
                            mmd_eigval = eigval_stats(true_eigvals, fake_eigvals, max_eig=self.trainer.datamodule.max_k_eigval)
                            emd_eigval = eigval_stats(true_eigvals, fake_eigvals, max_eig=self.trainer.datamodule.max_k_eigval, compute_emd=True)
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/mmd_eigval', mmd_eigval)
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/emd_eigval', emd_eigval)

                            if self.hparams.compute_emd:
                                measure_types =  ['mmd', 'emd']
                            else:
                                measure_types =  ['mmd']
                            for measure_type in measure_types:
                                mmd_degree = degree_stats(true_graphs, fake_graphs, compute_emd=(measure_type=='emd'))
                                mmd_4orbits = orbit_stats_all(true_graphs, fake_graphs, compute_emd=(measure_type=='emd'))
                                mmd_clustering = clustering_stats(true_graphs, fake_graphs, compute_emd=(measure_type=='emd'))    
                                mmd_spectral = spectral_stats(true_graphs, fake_graphs, compute_emd=(measure_type=='emd'))
                                mmd_spectral_k = spectral_stats(true_graphs, fake_graphs, n_eigvals=self.hparams.k_eigval, compute_emd=(measure_type=='emd'))
                                self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_degree', mmd_degree)
                                self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_4orbits', mmd_4orbits)
                                self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_clustering', mmd_clustering)
                                self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_spectral', mmd_spectral)
                                self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_spectral_k', mmd_spectral_k)

                                true_eigvecs = [g for x in outputs_i for g in x[f'{test_type}_true_eigvecs{model_type}']]
                                fake_eigvecs = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvecs{model_type}']] 
                                mmd_wavelet_eigvec = spectral_filter_stats(true_eigvecs, true_eigvals, fake_eigvecs, fake_eigvals, compute_emd=(measure_type=='emd'))
                                self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_spectral_filter', mmd_wavelet_eigvec)

                                true_graph_eigvals, true_graph_eigvecs = compute_list_eigh(true_graphs)
                                fake_graph_eigvals, fake_graph_eigvecs = compute_list_eigh(fake_graphs)
                                mmd_wavelet = spectral_filter_stats(true_graph_eigvecs, true_graph_eigvals, fake_graph_eigvecs, fake_graph_eigvals, compute_emd=(measure_type=='emd'))
                                self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_spectral_filter_graph', mmd_wavelet)

                            # Do graph validity tests
                            if 'lobster' in self.logger.log_dir:
                                acc = eval_acc_lobster_graph(fake_graphs)
                                self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/accuracy', acc)
                            elif 'tree' in self.logger.log_dir:
                                acc = eval_acc_tree_graph(fake_graphs)
                                self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/accuracy', acc)
                            elif 'grid' in self.logger.log_dir:
                                acc = eval_acc_grid_graph(fake_graphs)
                                self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/accuracy', acc)
                            elif 'sbm' in self.logger.log_dir:
                                acc = eval_acc_sbm_graph(fake_graphs, refinement_steps=100, strict=True)
                                self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/accuracy', acc)
                            elif 'planar' in self.logger.log_dir:
                                acc = eval_acc_planar_graph(fake_graphs)
                                self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/accuracy', acc)
                            else:
                                acc = 1.0

                            frac_unique = eval_fraction_unique(fake_graphs, precise=self.hparams.precise_uniqueness_val)
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/frac_unique', frac_unique)
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/frac_unique_x_accuracy', frac_unique * acc)

                            mean_MMD_ratio_3 = ((mmd_degree / (self.trainer.datamodule.train_mmd_degree + 1e-6)) +
                                        (mmd_4orbits / (self.trainer.datamodule.train_mmd_4orbits+ 1e-6)) + 
                                        (mmd_clustering / (self.trainer.datamodule.train_mmd_clustering + 1e-6))) / 3
                            mean_MMD_ratio_4 = ((mmd_degree / (self.trainer.datamodule.train_mmd_degree + 1e-6)) +
                                                (mmd_4orbits / (self.trainer.datamodule.train_mmd_4orbits+ 1e-6)) + 
                                                (mmd_clustering / (self.trainer.datamodule.train_mmd_clustering + 1e-6)) + 
                                                (mmd_spectral / (self.trainer.datamodule.train_mmd_spectral+ 1e-6))) / 4
                            mean_MMD_ratio_5 = ((mmd_degree / (self.trainer.datamodule.train_mmd_degree + 1e-6)) +
                                                (mmd_4orbits / (self.trainer.datamodule.train_mmd_4orbits+ 1e-6)) + 
                                                (mmd_clustering / (self.trainer.datamodule.train_mmd_clustering + 1e-6)) + 
                                                (mmd_spectral / (self.trainer.datamodule.train_mmd_spectral+ 1e-6)) + 
                                                (mmd_wavelet / (self.trainer.datamodule.train_mmd_wavelet+ 1e-6))) / 5
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/mean_MMD_ratio_3', mean_MMD_ratio_3)
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/mean_MMD_ratio_4', mean_MMD_ratio_4)
                            self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/mean_MMD_ratio_5', mean_MMD_ratio_5)

                        spectral_loss = torch.stack([x[f'{test_type}_spectral_loss{model_type}'] for x in outputs_i])
                        spectral_loss = spectral_loss[spectral_loss>0].mean().item()
                        normalized_spectral_loss = torch.stack([x[f'{test_type}_normalized_spectral_loss{model_type}'] for x in outputs_i])
                        normalized_spectral_loss = normalized_spectral_loss[normalized_spectral_loss>0].mean().item()
                        if cond_data_sufix == '':
                            recon_losses.append(spectral_loss)
                            normalized_recon_losses.append(normalized_spectral_loss)
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/spectrum_reconstruction_loss', spectral_loss)
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/normalized_spectrum_reconstruction_loss', normalized_spectral_loss)
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/spectral_decomp_fail', torch.stack([x[f'{test_type}_spectral_fail{model_type}'] for x in outputs_i]).mean())
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/spectral_match_loss', torch.stack([x[f'{test_type}_spectral_match_loss{model_type}'] for x in outputs_i]).mean())
                        self.log(f'validation_loss_{test_type}{cond_data_sufix}{model_type}/normalized_spectral_match_loss', torch.stack([x[f'{test_type}_normalized_spectral_match_loss{model_type}'] for x in outputs_i]).mean())
            self.log(f'validation_loss/mean_recon_loss', sum(recon_losses) / len(recon_losses))
            self.log(f'validation_loss/mean_normalized_recon_loss', sum(normalized_recon_losses) / len(normalized_recon_losses))

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        out_dict = {}
        # Run test on EMA params and current step params
        if self.hparams.ema > 0.0:
            with self.gen_ema.average_parameters():
                out_dict = out_dict | self.test_function(batch, batch_idx, dataloader_idx=dataloader_idx, out_dict=out_dict, suffix='_ema')
        out_dict = out_dict | self.test_function(batch, batch_idx, dataloader_idx=dataloader_idx, out_dict=out_dict, suffix='')

        test_loss = -self.current_epoch #decreases so that lightining saves checkpoint every val_step
        out_dict = {**out_dict, f'test_loss': test_loss}
        return out_dict

    def test_function(self, batch, batch_idx, dataloader_idx=0, out_dict={}, suffix=''):
        if dataloader_idx == 1:
            cond_data_sufix = '_train_cond'
        else:
            cond_data_sufix = ''
        Path(f"{self.logger.log_dir}/test/all_fake{cond_data_sufix}{suffix}/eigvecs").mkdir(parents=True, exist_ok=True)
        Path(f"{self.logger.log_dir}/test/fake_eigvec{cond_data_sufix}{suffix}/eigvecs").mkdir(parents=True, exist_ok=True)
        Path(f"{self.logger.log_dir}/test/fake_adj{cond_data_sufix}{suffix}/eigvecs").mkdir(parents=True, exist_ok=True)
        batch_max_n = max(batch['n_nodes'].max(), self.hparams.k_eigval)
        n_nodes = batch['n_nodes']
        real_eigval = batch["eigval"][:, :self.hparams.k_eigval]
        real_eigvec = batch["eigvec"][:, :batch_max_n, :self.hparams.k_eigval]
        mask = batch["mask"][:, :batch_max_n, :batch_max_n]
        adj_true = batch["adj"][:, :batch_max_n, :batch_max_n]

        cmap = cm.get_cmap('rainbow', self.hparams.k_eigval) 

        with torch.no_grad():
            if self.hparams.lambda_only:
                for test_type in ['all_fake']:
                    true_eigvals = []
                    fake_eigvals = []

                    t0 = time.perf_counter()
                    fake_eigval = self._get_fake(real_eigval, real_eigvec, mask, test_type=test_type)
                    torch.cuda.synchronize()
                    forward_pass_time = time.perf_counter() - t0

                    for i in range(real_eigval.size(0)):
                        true_eigvals.append(real_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvals.append(fake_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())

                    if batch_idx==0:
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                        ax.plot(real_eigval[0].cpu().detach().numpy())
                        ax.plot(fake_eigval[0].cpu().detach().numpy())
                        self.logger.experiment.add_figure(f'generated_eigvals_{test_type}', fig, self.current_epoch)
                        plt.close(fig)

                        plt.close('all')
                    out_dict = {**out_dict, f'{test_type}_true_eigvals{suffix}': true_eigvals, f'{test_type}_fake_eigvals{suffix}': fake_eigvals, f'{test_type}_forward_pass_time{suffix}': forward_pass_time}
            elif self.hparams.SON_only:
                for test_type in ['fake_eigvec']:
                    true_eigvals = []
                    fake_eigvals = []
                    true_eigvecs = []
                    fake_eigvecs = []

                    fake_eigvec, SON_aux_loss = self._get_fake(real_eigval, real_eigvec, mask, test_type=test_type)

                    for i in range(real_eigval.size(0)):
                        true_eigvals.append(real_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvals.append(real_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        true_eigvecs.append(real_eigvec[i, :n_nodes[i], :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvecs.append(fake_eigvec[i, :n_nodes[i], :self.hparams.k_eigval].cpu().detach().numpy())

                        eigvec_path = f"{self.logger.log_dir}/test/{test_type}{cond_data_sufix}{suffix}/eigvecs/graph_{batch_idx}_{i}.png"

                        # Plot true and fake eigvectors
                        fig, axs = plt.subplots(3, 1)
                        
                        axs[0].plot(real_eigval[i].cpu().detach().numpy())
                        axs[0].set_title('First k+1 eigvalues')

                        axs[1].plot(real_eigvec[i,:,1].cpu().detach().numpy())
                        axs[1].plot(fake_eigvec[i,:,1].cpu().detach().numpy())
                        axs[1].set_title('1st eigvector')
                        
                        axs[2].plot(real_eigvec[i,:,2].cpu().detach().numpy())
                        axs[2].plot(fake_eigvec[i,:,2].cpu().detach().numpy())
                        axs[2].set_title('2nd eigvector')

                        fig.tight_layout()
                        fig.savefig(eigvec_path)
                        plt.close(fig)

                    SON_loss = fake_eigvec @ real_eigvec.transpose(-2,-1)
                    SON_loss = SON_loss - torch.eye(SON_loss.size(1), SON_loss.size(2), device=SON_loss.device).unsqueeze(0)
                    SON_loss = torch.mean(SON_loss ** 2)

                    out_dict = {**out_dict, f'{test_type}_SON_loss{suffix}': SON_loss, f'{test_type}_true_eigvals{suffix}': true_eigvals, f'{test_type}_fake_eigvals{suffix}': fake_eigvals,
                                f'{test_type}_true_eigvecs{suffix}': true_eigvecs, f'{test_type}_fake_eigvecs{suffix}': fake_eigvecs}
            elif self.hparams.lambda_SON_only:
                for test_type in ['all_fake', 'fake_eigvec']:
                    true_eigvals = []
                    fake_eigvals = []
                    true_eigvecs = []
                    fake_eigvecs = []

                    fake_eigvec, mixed_eigval, fake_eigval, SON_aux_loss = self._get_fake(real_eigval, real_eigvec, mask, test_type=test_type)

                    for i in range(real_eigval.size(0)):
                        true_eigvals.append(real_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvals.append(mixed_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        true_eigvecs.append(real_eigvec[i, :n_nodes[i], :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvecs.append(fake_eigvec[i, :n_nodes[i], :self.hparams.k_eigval].cpu().detach().numpy())

                        eigvec_path = f"{self.logger.log_dir}/test/{test_type}{cond_data_sufix}{suffix}/eigvecs/graph_{batch_idx}_{i}.png"

                        # Plot true and fake eigvectors
                        fig, axs = plt.subplots(3, 1)
                        
                        axs[0].plot(real_eigval[i].cpu().detach().numpy())
                        axs[0].plot(fake_eigval[i].cpu().detach().numpy())
                        axs[0].set_title('First k+1 eigvalues')

                        axs[1].plot(real_eigvec[i,:,1].cpu().detach().numpy())
                        axs[1].plot(fake_eigvec[i,:,1].cpu().detach().numpy())
                        axs[1].set_title('1st eigvector')
                        
                        axs[2].plot(real_eigvec[i,:,2].cpu().detach().numpy())
                        axs[2].plot(fake_eigvec[i,:,2].cpu().detach().numpy())
                        axs[2].set_title('2nd eigvector')

                        fig.tight_layout()
                        fig.savefig(eigvec_path)
                        plt.close(fig)

                    SON_loss = fake_eigvec @ real_eigvec.transpose(-2,-1)
                    SON_loss = SON_loss - torch.eye(SON_loss.size(1), SON_loss.size(2), device=SON_loss.device).unsqueeze(0)
                    SON_loss = torch.mean(SON_loss ** 2)

                    out_dict = {**out_dict, f'{test_type}_SON_loss{suffix}': SON_loss, f'{test_type}_true_eigvals{suffix}': true_eigvals, f'{test_type}_fake_eigvals{suffix}': fake_eigvals,
                                f'{test_type}_true_eigvecs{suffix}': true_eigvecs, f'{test_type}_fake_eigvecs{suffix}': fake_eigvecs}

            else:
                if self.hparams.adj_only:
                    test_types = ['fake_adj']
                elif self.hparams.adj_eigvec_only:
                    test_types = ['fake_eigvec', 'fake_adj']
                else:
                    test_types = ['all_fake', 'fake_eigvec', 'fake_adj']
                for test_type in test_types:
                    true_graphs = []
                    fake_graphs = []
                    true_eigvals = []
                    fake_eigvals = []
                    true_eigvecs = []
                    fake_eigvecs = []
                    fake_node_features = []
                    fake_edge_features = []
                    fake_adj = []
                    t0 = time.perf_counter()
                    adj, node_features, edge_features, mixed_eigvec, fake_eigvec, mixed_eigval, fake_eigval, mixed_eigvec_eigval, SON_aux_loss, adj_noise = self._get_fake(real_eigval, real_eigvec, mask, test_type=test_type, return_adj_noise=True)
                    torch.cuda.synchronize()
                    forward_pass_time = time.perf_counter() - t0

                    # Discretize adjacency:
                    adj[adj < 0] = 0.0 
                    adj = zero_diag(torch.round(adj))
                    adj[adj > 1] = 1.0 

                    if self.hparams.ignore_first_eigv:
                        eigv_offset = 1
                    else:
                        eigv_offset = 0

                    if self.hparams.qm9:
                        # Discretize node and edge classes:
                        node_features = torch.argmax(node_features, dim=-1)
                        # Only keep edge features for existing edges
                        edge_features = (torch.argmax(edge_features, dim=-1) + 1) * adj # +1 for conversion to weighted nx matrix

                    for i in range(adj.size(0)):
                        true_eigvals.append(real_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvals.append(mixed_eigval[i, :self.hparams.k_eigval].cpu().detach().numpy())
                        true_eigvecs.append(real_eigvec[i, :n_nodes[i], :self.hparams.k_eigval].cpu().detach().numpy())
                        fake_eigvecs.append(fake_eigvec[i, :n_nodes[i], :self.hparams.k_eigval].cpu().detach().numpy())
                        if self.hparams.qm9:
                            #  Add edge features as edege "weight"
                            nx_graph = nx.from_numpy_array(edge_features[i, :n_nodes[i], :n_nodes[i]].cpu().detach().numpy())
                            #  Add node features  to nx graph
                            for j in range(n_nodes[i].item()):
                                nx_graph.nodes[j]["class"] = node_features[i, j].item()
                            fake_graphs.append(nx_graph)
                            fake_node_features.append(node_features[i, :n_nodes[i]].cpu().detach().long())
                            fake_edge_features.append((edge_features[i, :n_nodes[i], :n_nodes[i]] - 1).cpu().detach().long()) #Start edge class indexing from 0 for BasicMolecularMetrics
                            fake_adj.append(adj[i, :n_nodes[i], :n_nodes[i]].cpu().detach().long())
                            out_dict = {**out_dict, f'{test_type}_fake_node_features{suffix}': fake_node_features, f'{test_type}_fake_edge_features{suffix}': fake_edge_features, f'{test_type}_fake_adj{suffix}': fake_adj}
                        else:
                            true_graphs.append(nx.from_numpy_array(adj_true[i, :n_nodes[i], :n_nodes[i]].cpu().detach().numpy()))
                            fake_graphs.append(nx.from_numpy_array(adj[i, :n_nodes[i], :n_nodes[i]].cpu().detach().numpy()))

                            path = f"{self.logger.log_dir}/test/{test_type}{cond_data_sufix}{suffix}/graph_{batch_idx}_{i}.png"

                            fig, axs = plt.subplots(2, 2)

                            im0 = axs[0, 0].imshow(adj[i].cpu().detach().numpy(), interpolation='none')
                            divider = make_axes_locatable(axs[0, 0])
                            cax = divider.append_axes('right', size='5%', pad=0.05)
                            fig.colorbar(im0, cax=cax, orientation='vertical')
                            axs[0, 0].set_title('Fake')

                            G = nx.convert_matrix.from_numpy_array(adj[i, :n_nodes[i], :n_nodes[i]].cpu().detach().numpy())
                            pos = nx.spring_layout(G, iterations=50)
                            kmeans = KMeans(n_clusters=min(self.hparams.k_eigval, n_nodes[i].item()), n_init=100).fit(mixed_eigvec[i].cpu().detach().numpy()[:n_nodes[i],:self.hparams.k_eigval])
                            y_pred = kmeans.labels_
                            nx.draw(G, pos, font_size=5, node_size=12, with_labels=True, node_color=cmap(y_pred), ax=axs[1, 0])
                            
                            im1 = axs[0, 1].imshow(adj_true[i].cpu().detach().numpy(), interpolation='none')
                            divider = make_axes_locatable(axs[0, 1])
                            cax = divider.append_axes('right', size='5%', pad=0.05)
                            fig.colorbar(im1, cax=cax, orientation='vertical')
                            axs[0, 1].set_title('True')

                            G = nx.convert_matrix.from_numpy_array(adj_true[i, :n_nodes[i], :n_nodes[i]].cpu().detach().numpy())
                            pos = nx.spring_layout(G, iterations=50)
                            kmeans = KMeans(n_clusters=min(self.hparams.k_eigval, n_nodes[i].item()), n_init=100).fit(real_eigvec[i].cpu().detach().numpy()[:n_nodes[i],:self.hparams.k_eigval])
                            y_pred = kmeans.labels_
                            nx.draw(G, pos, font_size=5, node_size=12, with_labels=True, node_color=cmap(y_pred), ax=axs[1, 1])

                            fig.tight_layout()
                            fig.savefig(path)
                            plt.close(fig)

                            deg = adj[i].sum(dim=-1)
                            D = torch.diag_embed(deg.abs()**(-0.5))
                            L = torch.eye(adj.size(1), adj.size(2), out=torch.zeros_like(adj[i])) - D @ adj[i] @ D
                            L[L.isnan()] = 0
                            del D
                            L = L[:n_nodes[i], :n_nodes[i]]

                            try:
                                eigvec_path = f"{self.logger.log_dir}/test/{test_type}{cond_data_sufix}{suffix}/eigvecs/graph_{batch_idx}_{i}.png"
                                rec_eigval, rec_eigvec = torch.linalg.eigh(L)

                                # Plot true and fake eigvectors
                                fig, axs = plt.subplots(3, 1)
                                
                                axs[0].plot(real_eigval[i].cpu().detach().numpy())
                                axs[0].plot(rec_eigval[:self.hparams.k_eigval].cpu().detach().numpy())
                                axs[0].plot(fake_eigval[i].cpu().detach().numpy())
                                axs[0].set_title('First k+1 eigvalues')

                                axs[1].plot(real_eigvec[i,:,0].cpu().detach().numpy())
                                axs[1].plot(rec_eigvec[:,0+eigv_offset].cpu().detach().numpy())
                                axs[1].plot(mixed_eigvec[i,:,0].cpu().detach().numpy())
                                axs[1].set_title('1st eigvector')
                                
                                axs[2].plot(real_eigvec[i,:,1].cpu().detach().numpy())
                                axs[2].plot(rec_eigvec[:,1+eigv_offset].cpu().detach().numpy())
                                axs[2].plot(mixed_eigvec[i,:,1].cpu().detach().numpy())
                                axs[2].set_title('2nd eigvector')

                                fig.tight_layout()
                                fig.savefig(eigvec_path)
                                plt.close(fig)
                            except RuntimeError:
                                pass

                            plt.close('all')

                        # Compute L2 distance between input eigvalues and eigvalues of discretized generated adjacency matrix
                        deg = adj.sum(dim=-1)
                        D = torch.diag_embed(deg.abs()**(-0.5))
                        L = torch.eye(adj.size(1), adj.size(2), out=torch.zeros_like(adj)) - D @ adj @ D
                        L[L.isnan()] = 0
                        del D
                        spectral_fail = torch.tensor(0.0, device=self.device)
                        spectral_loss = -torch.ones(L.size(0), device=L.device)
                        normalized_spectral_loss = -torch.ones(L.size(0), device=L.device)
                        for i in range(L.size(0)):
                            try:
                                rec_eigval = torch.linalg.eigvalsh(L[i, :n_nodes[i], :n_nodes[i]])[eigv_offset:]
                                spectral_loss[i] = torch.mean((mixed_eigvec_eigval[i, :min(n_nodes[i].item(), self.hparams.k_eigval)][mixed_eigvec_eigval[i, :min(n_nodes[i].item(), self.hparams.k_eigval)] > 0] - rec_eigval[:self.hparams.k_eigval][mixed_eigvec_eigval[i, :min(n_nodes[i].item(), self.hparams.k_eigval)] > 0])**2)
                                normalized_spectral_loss[i] = torch.mean(torch.abs((mixed_eigvec_eigval[i, :self.hparams.k_eigval] - rec_eigval[:self.hparams.k_eigval])[mixed_eigvec_eigval[i, :self.hparams.k_eigval] > 0] / mixed_eigvec_eigval[i, :self.hparams.k_eigval][mixed_eigvec_eigval[i, :self.hparams.k_eigval] > 0]))
                            except RuntimeError:
                                spectral_fail += 1.0
                        spectral_fail = spectral_fail / L.size(0)
                        spectral_loss = spectral_loss.mean()
                        normalized_spectral_loss = normalized_spectral_loss.mean()
                        # u^T @ L @ u - lambda - take only the number of eigvectors used to condition the model
                        spectral_match_loss = torch.tensor(0.0, device=self.device)
                        normalized_spectral_match_loss = torch.tensor(0.0, device=self.device)
                        for i in range(0, self.hparams.k_eigval):
                            spectral_match = (mixed_eigvec[:, :, i].view(mixed_eigvec.size(0),1,-1) @ L @ mixed_eigvec[:, :, i].view(mixed_eigvec.size(0),-1,1) - mixed_eigvec_eigval[:,i].view(mixed_eigvec_eigval.size(0),1,1))              
                            spectral_match_loss += torch.mean(spectral_match**2) / self.hparams.k_eigval
                            normalized_spectral_match_loss += ( torch.mean(spectral_match.abs()[mixed_eigvec_eigval[:,i]>0] / mixed_eigvec_eigval[:,i][mixed_eigvec_eigval[:,i]>0].view(mixed_eigvec_eigval[:,i][mixed_eigvec_eigval[:,i]>0].size(0),1,1)) ) / self.hparams.k_eigval
                        
                        out_dict = {**out_dict, f'{test_type}_true_graphs{suffix}': true_graphs, f'{test_type}_fake_graphs{suffix}': fake_graphs, f'{test_type}_true_eigvals{suffix}': true_eigvals, f'{test_type}_fake_eigvals{suffix}': fake_eigvals,
                                                f'{test_type}_spectral_loss{suffix}': spectral_loss, f'{test_type}_normalized_spectral_loss{suffix}': normalized_spectral_loss, 
                                                f'{test_type}_spectral_match_loss{suffix}': spectral_match_loss, f'{test_type}_normalized_spectral_match_loss{suffix}': normalized_spectral_match_loss, f'{test_type}_spectral_fail{suffix}': spectral_fail,
                                                f'{test_type}_forward_pass_time{suffix}': forward_pass_time, f'{test_type}_true_eigvecs{suffix}': true_eigvecs, f'{test_type}_fake_eigvecs{suffix}': fake_eigvecs}  

        return out_dict

    def test_epoch_end(self, outputs):
        if type(outputs[0]) == list:
            cond_data = ['', '_train_cond']
            single = False
        else:
            cond_data = ['']
            single = True
        if self.hparams.ema > 0.0:
            model_types = ['', '_ema']
        else:
            model_types = ['']
        for model_type in model_types:
            for dataloader_idx, cond_data_sufix in enumerate(cond_data):
                if not single:
                    outputs_i = outputs[dataloader_idx]
                else:
                    outputs_i = outputs
                if self.hparams.lambda_only:
                    for test_type in ['all_fake']:
                        true_eigvals = [g for x in outputs_i for g in x[f'{test_type}_true_eigvals{model_type}']]
                        fake_eigvals = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvals{model_type}']] 
                        mmd_eigval = eigval_stats(true_eigvals, fake_eigvals, max_eig=self.trainer.datamodule.max_k_eigval)
                        emd_eigval = eigval_stats(true_eigvals, fake_eigvals, max_eig=self.trainer.datamodule.max_k_eigval, compute_emd=True)
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/mmd_eigval', mmd_eigval)
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/emd_eigval', emd_eigval)
                elif self.hparams.SON_only:
                    test_types = ['fake_eigvec']
                    for test_type in test_types:
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/SON_loss', torch.stack([x[f'{test_type}_SON_loss{model_type}'] for x in outputs_i]).mean())

                        true_eigvals = [g for x in outputs_i for g in x[f'{test_type}_true_eigvals{model_type}']]
                        fake_eigvals = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvals{model_type}']] 

                        true_eigvecs = [g for x in outputs_i for g in x[f'{test_type}_true_eigvecs{model_type}']]
                        fake_eigvecs = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvecs{model_type}']] 
                        if self.hparams.compute_emd:
                            measure_types =  ['mmd', 'emd']
                        else:
                            measure_types =  ['mmd']
                        for measure_type in measure_types:
                            mmd_wavelet_eigvec = spectral_filter_stats(true_eigvecs, true_eigvals, fake_eigvecs, fake_eigvals, compute_emd=(measure_type=='emd'))
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_spectral_filter', mmd_wavelet_eigvec)
                elif self.hparams.lambda_SON_only:
                    test_types = ['all_fake', 'fake_eigvec']
                    for test_type in test_types:
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/SON_loss', torch.stack([x[f'{test_type}_SON_loss{model_type}'] for x in outputs_i]).mean())

                        true_eigvals = [g for x in outputs_i for g in x[f'{test_type}_true_eigvals{model_type}']]
                        fake_eigvals = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvals{model_type}']] 

                        true_eigvecs = [g for x in outputs_i for g in x[f'{test_type}_true_eigvecs{model_type}']]
                        fake_eigvecs = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvecs{model_type}']] 

                        mmd_eigval = eigval_stats(true_eigvals, fake_eigvals, max_eig=self.trainer.datamodule.max_k_eigval)
                        emd_eigval = eigval_stats(true_eigvals, fake_eigvals, max_eig=self.trainer.datamodule.max_k_eigval, compute_emd=True)
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/mmd_eigval', mmd_eigval)
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/emd_eigval', emd_eigval)
                        
                        mean_MMD_ratio = (mmd_eigval / (self.trainer.datamodule.train_mmd_spectral+ 1e-6))
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/mean_MMD_ratio', mean_MMD_ratio)
                        if self.hparams.compute_emd:
                            measure_types =  ['mmd', 'emd']
                        else:
                            measure_types =  ['mmd']
                        for measure_type in measure_types:
                            mmd_wavelet_eigvec = spectral_filter_stats(true_eigvecs, true_eigvals, fake_eigvecs, fake_eigvals, compute_emd=(measure_type=='emd'))
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_spectral_filter', mmd_wavelet_eigvec)
                else:
                    if self.hparams.adj_only:
                        test_types = ['fake_adj']
                    elif self.hparams.adj_eigvec_only:
                        test_types = ['fake_eigvec', 'fake_adj']
                    elif self.hparams.SON_only:
                        test_types = ['fake_eigvec']
                    else:
                        test_types = ['all_fake', 'fake_eigvec', 'fake_adj']
                    for test_type in test_types:
                        true_graphs = [nx.from_numpy_array(graph["adj"][:graph['n_nodes'], :graph['n_nodes']].cpu().detach().numpy()) for graph in self.trainer.datamodule.test] # Always compare to test set
                        fake_graphs = [g for x in outputs_i for g in x[f'{test_type}_fake_graphs{model_type}']] 
                        true_eigvals = [graph["eigval"][:self.hparams.k_eigval].cpu().detach().numpy() for graph in self.trainer.datamodule.test] # Always compare to test set
                        fake_eigvals = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvals{model_type}']] 
                        true_eigvecs = [graph["eigvec"][:, :self.hparams.k_eigval].cpu().detach().numpy() for graph in self.trainer.datamodule.test]
                        fake_eigvecs = [g for x in outputs_i for g in x[f'{test_type}_fake_eigvecs{model_type}']]

                        torch.save(fake_graphs, f"{self.logger.log_dir}/test/{test_type}{cond_data_sufix}{model_type}/generated_graphs.pt")
                        torch.save(fake_eigvals, f"{self.logger.log_dir}/test/{test_type}{cond_data_sufix}{model_type}/generated_graphs_eigval.pt")
                        torch.save(fake_eigvecs, f"{self.logger.log_dir}/test/{test_type}{cond_data_sufix}{model_type}/generated_graphs_eigvec.pt")
                        torch.save(true_graphs, f"{self.logger.log_dir}/test/true_graphs.pt")
                        torch.save(true_eigvals, f"{self.logger.log_dir}/test/true_graphs_eigval.pt")
                        torch.save(true_eigvecs, f"{self.logger.log_dir}/test/true_graphs_eigvec.pt")
                        if self.hparams.use_fixed_emb:
                            torch.save(self.generator.embedding.weight.data, f"{self.logger.log_dir}/test/{test_type}{cond_data_sufix}{model_type}/fixed_emb.pt")

                        if self.hparams.qm9:
                            fake_node_features = [g for x in outputs_i for g in x[f'{test_type}_fake_node_features{model_type}']] 
                            fake_edge_features = [g for x in outputs_i for g in x[f'{test_type}_fake_edge_features{model_type}']] 
                            fake_adj = [g for x in outputs_i for g in x[f'{test_type}_fake_adj{model_type}']] 
                            molecule_metrics, _ =  self.trainer.datamodule.molecular_metrics.evaluate(list(zip(fake_node_features, fake_adj, fake_edge_features)), filename=f"{self.logger.log_dir}/test/{test_type}{cond_data_sufix}{model_type}/gen_mol")
                            valid, unique, novel = molecule_metrics
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/frac_valid', valid)
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/frac_unique', unique)
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/frac_novel', novel)
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/valid_and_unique', valid*unique) # unique is returned as fraction of valid
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/valid_and_unique_and_novel', valid*unique*novel) # unique is returned as fraction of valid
                        else:
                            # Get MMD measures
                            mmd_eigval = eigval_stats(true_eigvals, fake_eigvals, max_eig=self.trainer.datamodule.max_k_eigval)
                            emd_eigval = eigval_stats(true_eigvals, fake_eigvals, max_eig=self.trainer.datamodule.max_k_eigval, compute_emd=True)
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/mmd_eigval', mmd_eigval)
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/emd_eigval', emd_eigval)

                            if self.hparams.compute_emd:
                                measure_types =  ['mmd', 'emd']
                            else:
                                measure_types =  ['mmd']
                            for measure_type in measure_types:
                                mmd_degree = degree_stats(true_graphs, fake_graphs, compute_emd=(measure_type=='emd'))
                                mmd_4orbits = orbit_stats_all(true_graphs, fake_graphs, compute_emd=(measure_type=='emd'))
                                mmd_clustering = clustering_stats(true_graphs, fake_graphs, compute_emd=(measure_type=='emd'))    
                                mmd_spectral = spectral_stats(true_graphs, fake_graphs, compute_emd=(measure_type=='emd'))
                                mmd_spectral_k = spectral_stats(true_graphs, fake_graphs, n_eigvals=self.hparams.k_eigval, compute_emd=(measure_type=='emd'))
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_degree', mmd_degree)
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_4orbits', mmd_4orbits)
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_clustering', mmd_clustering)
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_spectral', mmd_spectral)
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_spectral_k', mmd_spectral_k)

                                mmd_wavelet_eigvec = spectral_filter_stats(true_eigvecs, true_eigvals, fake_eigvecs, fake_eigvals, compute_emd=(measure_type=='emd'))
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_spectral_filter', mmd_wavelet_eigvec)

                                true_graph_eigvals, true_graph_eigvecs = compute_list_eigh(true_graphs)
                                fake_graph_eigvals, fake_graph_eigvecs = compute_list_eigh(fake_graphs)
                                mmd_wavelet = spectral_filter_stats(true_graph_eigvecs, true_graph_eigvals, fake_graph_eigvecs, fake_graph_eigvals, compute_emd=(measure_type=='emd'))
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/{measure_type}_spectral_filter_graph', mmd_wavelet)

                            # Do graph validity tests
                            if 'lobster' in self.logger.log_dir:
                                acc = eval_acc_lobster_graph(fake_graphs)
                                validity_func = is_lobster_graph
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/accuracy', acc)
                            elif 'tree' in self.logger.log_dir:
                                acc = eval_acc_tree_graph(fake_graphs)
                                validity_func = nx.is_tree
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/accuracy', acc)
                            elif 'grid' in self.logger.log_dir:
                                acc = eval_acc_grid_graph(fake_graphs)
                                validity_func = is_grid_graph
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/accuracy', acc)
                            elif 'sbm' in self.logger.log_dir:
                                acc = eval_acc_sbm_graph(fake_graphs, refinement_steps=1000, strict=False)
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/accuracy', acc)
                                acc_strict = eval_acc_sbm_graph(fake_graphs, refinement_steps=1000)
                                validity_func = is_sbm_graph
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/accuracy_strict', acc_strict)
                            elif 'qm9' in self.logger.log_dir:
                                raise NotImplemented
                            elif 'planar' in self.logger.log_dir:
                                acc = eval_acc_planar_graph(fake_graphs)
                                validity_func = is_planar_graph
                                self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/accuracy', acc)
                            else:
                                validity_func = lambda x: True

                            # Check how many generated graphs are unique and how many of those are non-isomorphic to the train set:
                            if test_type == 'fake_adj':
                                # When we condition on true eigenvectors and eigenvalues from the test set compare uniqueness to test graphs
                                train_graphs = [nx.from_numpy_array(graph["adj"][:graph['n_nodes'], :graph['n_nodes']].cpu().detach().numpy()) for graph in self.trainer.datamodule.test]
                            else:
                                train_graphs = [nx.from_numpy_array(graph["adj"][:graph['n_nodes'], :graph['n_nodes']].cpu().detach().numpy()) for graph in self.trainer.datamodule.train]
                            frac_unique, frac_unique_non_isomorphic, fraction_unique_non_isomorphic_valid = eval_fraction_unique_non_isomorphic_valid(fake_graphs, train_graphs, validity_func)
                            frac_non_isomorphic = 1.0 - eval_fraction_isomorphic(fake_graphs, train_graphs)
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/frac_unique', frac_unique)
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/frac_unique_non_isomorphic', frac_unique_non_isomorphic)
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/frac_unique_non_isomorphic_valid', fraction_unique_non_isomorphic_valid)
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/frac_non_isomorphic', frac_non_isomorphic)

                            # MMD Ratios
                            mean_MMD_ratio_3 = ((mmd_degree / (self.trainer.datamodule.train_mmd_degree + 1e-6)) +
                                        (mmd_4orbits / (self.trainer.datamodule.train_mmd_4orbits+ 1e-6)) + 
                                        (mmd_clustering / (self.trainer.datamodule.train_mmd_clustering + 1e-6))) / 3
                            mean_MMD_ratio_4 = ((mmd_degree / (self.trainer.datamodule.train_mmd_degree + 1e-6)) +
                                                (mmd_4orbits / (self.trainer.datamodule.train_mmd_4orbits+ 1e-6)) + 
                                                (mmd_clustering / (self.trainer.datamodule.train_mmd_clustering + 1e-6)) + 
                                                (mmd_spectral / (self.trainer.datamodule.train_mmd_spectral+ 1e-6))) / 4
                            mean_MMD_ratio_5 = ((mmd_degree / (self.trainer.datamodule.train_mmd_degree + 1e-6)) +
                                                (mmd_4orbits / (self.trainer.datamodule.train_mmd_4orbits+ 1e-6)) + 
                                                (mmd_clustering / (self.trainer.datamodule.train_mmd_clustering + 1e-6)) + 
                                                (mmd_spectral / (self.trainer.datamodule.train_mmd_spectral+ 1e-6)) + 
                                                (mmd_wavelet / (self.trainer.datamodule.train_mmd_wavelet+ 1e-6))) / 5
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/mean_MMD_ratio_3', mean_MMD_ratio_3)
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/mean_MMD_ratio_4', mean_MMD_ratio_4)
                            self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/mean_MMD_ratio_5', mean_MMD_ratio_5)

                        spectral_loss = torch.stack([x[f'{test_type}_spectral_loss{model_type}'] for x in outputs_i])
                        normalized_spectral_loss = torch.stack([x[f'{test_type}_normalized_spectral_loss{model_type}'] for x in outputs_i])
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/spectrum_reconstruction_loss', spectral_loss[spectral_loss>0].mean())
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/normalized_spectrum_reconstruction_loss', normalized_spectral_loss[normalized_spectral_loss>0].mean())
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/spectral_decomp_fail', torch.stack([x[f'{test_type}_spectral_fail{model_type}'] for x in outputs_i]).mean())
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/spectral_match_loss', torch.stack([x[f'{test_type}_spectral_match_loss{model_type}'] for x in outputs_i]).mean())
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/normalized_spectral_match_loss', torch.stack([x[f'{test_type}_normalized_spectral_match_loss{model_type}'] for x in outputs_i]).mean())

                        forward_pass_time = [x[f'{test_type}_forward_pass_time{model_type}'] for x in outputs_i]
                        forward_pass_time = sum(forward_pass_time) / len(forward_pass_time)
                        self.log(f'test_loss_{test_type}{cond_data_sufix}{model_type}/forward_pass_time', forward_pass_time)
                

    def configure_optimizers(self):
        lr_d = self.hparams.lr_d
        lr_g = self.hparams.lr_g
        betas = (self.hparams.beta1, self.hparams.beta2)
        opt_disc = torch.optim.Adam(list(self.discriminator.parameters()) + list(self.SON_discriminator.parameters()) + list(self.lambda_discriminator.parameters()), lr=lr_d, betas=betas, weight_decay=self.hparams.weight_decay)
        opt_gen = torch.optim.Adam(self.gen_params, lr=lr_g, betas=betas, weight_decay=self.hparams.weight_decay)
        return [opt_disc, opt_gen], [torch.optim.lr_scheduler.ExponentialLR(opt_disc, self.hparams.lr_D_decay), torch.optim.lr_scheduler.ExponentialLR(opt_gen, self.hparams.lr_G_decay)]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', default='', type=str)
    parser.add_argument('--log_grads', default=False, action="store_true")
    parser.add_argument('--disable_checkpoints', default=False, action="store_true")
    parser.add_argument('--job_id', default='', type=str)
    parser = GraphDataModule.add_data_specific_args(parser)
    parser = SPECTRE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    # Modify trainer defaults
    parser.set_defaults(max_epochs=1000, log_every_n_steps=5, check_val_every_n_epoch=5, gpus=1)
    # ------------------------------------------------------------------------
    # Set Defaults for Trainer params
    parser.set_defaults(batch_size=10)
    # ------------------------------------------------------------------------
    args = parser.parse_args()
    # Baselines only generate the adjacency
    args.adj_only = args.mlp_gen or args.no_cond or args.use_fixed_emb or args.adj_only

    pl.seed_everything(args.seed)

    data_module_params = {'batch_size': args.batch_size, 'k': args.k_eigval, 'n_nodes': args.n_nodes, 'n_graphs': args.n_graphs,
                        'n_data_workers': args.n_data_workers, 'same_sample': args.same_sample, 'n_start': args.n_start,
                        'n_end': args.n_end, 'dataset': args.dataset, 'validate_on_train_cond': args.validate_on_train_cond,
                        'ignore_first_eigv': args.ignore_first_eigv, 'qm9_strict_eval': args.qm9_strict_eval}
    data_module = GraphDataModule(**data_module_params)

    model_params = dict(beta1=args.beta1, beta2=args.beta2, lr_g=args.lr_g, lr_d=args.lr_d, gp_lambda=args.gp_lambda, n_max=args.n_max, 
                        gen_leaky_ReLU_alpha=args.gen_leaky_ReLU_alpha, disc_leaky_ReLU_alpha=args.disc_leaky_ReLU_alpha, n_G=args.n_G,
                        n_D=args.n_D, hid_G=args.hid_G, hid_D=args.hid_D, disc_normalization=args.disc_normalization, k_eigval=args.k_eigval,
                        gen_gelu=args.gen_gelu, disc_gelu=args.disc_gelu, disc_step_multiplier=args.disc_step_multiplier, weight_decay=args.weight_decay,
                        use_fixed_emb=args.use_fixed_emb, gen_normalization=args.gen_normalization, eigvec_temp_decay=args.eigvec_temp_decay,
                        decay_eigvec_temp_over=args.decay_eigvec_temp_over, spectral_norm=args.spectral_norm, G_dropout=args.G_dropout, D_dropout=args.D_dropout,
                        skip_connection=args.skip_connection, n_rot=args.n_rot, cat_eigvals=args.cat_eigvals, cat_mult_eigvals=args.cat_mult_eigvals,
                        disc_aux=args.disc_aux, n_eigval_warmup_epochs=args.n_eigval_warmup_epochs, n_eigvec_warmup_epochs=args.n_eigvec_warmup_epochs,
                        SON_disc=args.SON_disc, eigval_noise=args.eigval_noise, min_eigvec_temp=args.min_eigvec_temp, SON_max_pool=args.SON_max_pool,
                        SON_skip_connection=args.SON_skip_connection, SON_share_weights=args.SON_share_weights, SON_D_full_readout=args.SON_D_full_readout,
                        SON_D_n_rot=args.SON_D_n_rot, rand_rot_var=args.rand_rot_var, noise_latent_dim=args.noise_latent_dim, lambda_disc=args.lambda_disc,
                        eigval_temp_decay=args.eigval_temp_decay, decay_eigval_temp_over=args.decay_eigval_temp_over, min_eigval_temp=args.min_eigval_temp,
                        max_eigval_temp=args.max_eigval_temp, max_eigvec_temp=args.max_eigvec_temp, eigvec_noise=args.eigvec_noise, adj_noise=args.adj_noise,
                        edge_noise=args.edge_noise, edge_eigvecs=args.edge_eigvecs, lambda_only=args.lambda_only, lambda_norm=args.lambda_norm,
                        lambda_upsample=args.lambda_upsample, lr_decay_every=args.lr_decay_every, lr_decay_warmup=args.lr_decay_warmup, lr_D_decay=args.lr_D_decay,
                        lr_G_decay=args.lr_G_decay, adj_only=args.adj_only, adj_eigvec_only=args.adj_eigvec_only, SON_only=args.SON_only, lambda_SON_only=args.lambda_SON_only,
                        SON_normalize_left=args.SON_normalize_left, noisy_gen=args.noisy_gen, lambda_gating=args.lambda_gating,
                        lambda_last_gating=args.lambda_last_gating, lambda_last_linear=args.lambda_last_linear, lambda_dropout=args.lambda_dropout,
                        gp_adj_rewire=args.gp_adj_rewire, gp_adj_noise=args.gp_adj_noise,
                        wgan_eps=args.wgan_eps, ema=args.ema, compute_emd=args.compute_emd, SON_small=args.SON_small, temp_new=args.temp_new, pretrain=args.pretrain,
                        gp_do_backwards=args.gp_do_backwards, disc_noise_rewire=args.disc_noise_rewire, D_partial_laplacian=args.D_partial_laplacian,
                        derived_eigval_noise=args.derived_eigval_noise, normalize_noise=args.normalize_noise, noisy_disc=args.noisy_disc, SON_init_bank_size=args.SON_init_bank_size,
                        SON_gumbel_temperature=args.SON_gumbel_temperature, eigvec_right_noise=args.eigvec_right_noise, min_SON_gumbel_temperature=args.min_SON_gumbel_temperature,
                        SON_gumbel_temperature_decay=args.SON_gumbel_temperature_decay, decay_SON_gumbel_temp_over=args.decay_SON_gumbel_temp_over,
                        SON_gumbel_temperature_warmup_epochs=args.SON_gumbel_temperature_warmup_epochs, gp_shared_alpha=args.gp_shared_alpha, sharp_restart=args.sharp_restart,
                        no_restart=args.no_restart, precise_uniqueness_val=args.precise_uniqueness_val, SON_kl_init_scale=args.SON_kl_init_scale,
                        SON_stiefel_sim_init=args.SON_stiefel_sim_init, mlp_gen=args.mlp_gen, use_eigvecs=args.use_eigvecs, no_cond=args.no_cond, init_emb_channels=args.init_emb_channels,
                        eigvec_sign_flip=args.eigvec_sign_flip, ignore_first_eigv=args.ignore_first_eigv, gp_include_unpermuted=args.gp_include_unpermuted,
                        ppgn_data_channels_mult=args.ppgn_data_channels_mult, skip_noise_preprocess=args.skip_noise_preprocess, clip_grad_norm=args.clip_grad_norm, qm9=(args.dataset == 'qm9'))

    # Create custom name for the experiment
    start_time = time.strftime("%Y%m%d-%H%M%S")
    if len(args.job_id) > 1:
        rand_string = 'j'+args.job_id
    else:
        rand_string = ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=10))
    data_module.setup()

    model_params['n_max'] = data_module.n_max
    print(f'Largest graph has {data_module.n_max} nodes')
    model = SPECTRE(**model_params)
    

    if args.resume_from_checkpoint is not None:
        checkpoint_folder = os.path.dirname(os.path.dirname(args.resume_from_checkpoint))
        version_string = os.path.basename(os.path.normpath(checkpoint_folder))
    else:
        version_string = f"SPECTRE{'_fixed_emb' if args.use_fixed_emb else ''}{'_no_cond' if args.no_cond else ''}{'_mlp_gen' if args.mlp_gen else ''}{'_skip' if args.skip_connection else ''}_rot_{args.n_rot}{'_max' if args.SON_max_pool else ''}{'_sw' if args.SON_share_weights else ''}{'_sc' if args.SON_skip_connection else ''}_en_{args.eigval_noise}_{args.eigvec_noise}_{args.adj_noise}_beta_{args.beta1}_{args.beta2}_wd_{args.weight_decay}_k_{args.k_eigval}{f'_G_d_{args.G_dropout}' if args.G_dropout > 0.0 else ''}{f'_D_d_{args.G_dropout}' if args.D_dropout > 0.0 else ''}_{args.disc_normalization}_norm_D_{args.disc_step_multiplier}_n_G_{args.n_G}_D_{args.n_D}_hid_G_{args.hid_G}_D_{args.hid_D}{'_SN' if args.spectral_norm else ''}_{'GGELU' if args.gen_gelu else f'GLReLU_{args.gen_leaky_ReLU_alpha}'}_{'DGELU' if args.disc_gelu else f'DLReLU_{args.disc_leaky_ReLU_alpha}'}_bs_{args.batch_size}_lr_G_{args.lr_g}_D_{args.lr_d}{'_adj_only' if args.adj_only else ''}{'_adj_eigvec_only' if args.adj_eigvec_only else ''}_{data_module.dataset_string}_{start_time}_{rand_string}"
    
    # Setup loging
    logger = TensorBoardLogger(save_dir='logs', name=args.experiment_name, version=version_string) #version="Custom experiment version string"
    args.logger = logger
    callbacks = [
        ModelCheckpoint(
                save_on_train_epoch_end=True,
                save_last=True
            ),
    ]
    if not args.disable_checkpoints:
        callbacks += [
            ModelCheckpoint(
                monitor=f"validation_loss/mean_recon_loss",
                filename="best_mean_recon_{epoch:06d}-{validation_loss/mean_recon_loss:.4f}",
                save_top_k=1,
                mode="min",
                save_on_train_epoch_end=False,
                auto_insert_metric_name=False
            ),
            ModelCheckpoint(
                monitor=f"validation_loss/mean_normalized_recon_loss",
                filename="best_mean_normalized_recon_loss_{epoch:06d}-{validation_loss/mean_normalized_recon_loss:.4f}",
                save_top_k=1,
                mode="min",
                save_on_train_epoch_end=False,
                auto_insert_metric_name=False
            ),
        ]
        model_types = ['', '_ema']
        if args.adj_only:
            test_types = ['fake_adj']
        elif args.adj_eigvec_only:
            test_types = ['fake_eigvec', 'fake_adj']
        else:
            test_types = ['all_fake', 'fake_eigvec', 'fake_adj']
        for test_type in test_types:
            for model_type in model_types:
                if args.dataset == 'qm9':
                    callbacks += [
                        ModelCheckpoint(
                            monitor=f"validation_loss_{test_type}{model_type}/frac_valid",
                            filename=f"best_frac_valid_{test_type}{model_type}"+"_{epoch:06d}-{"+f"validation_loss_{test_type}{model_type}/frac_valid"+":.4f}",
                            save_top_k=1,
                            mode="max",
                            save_on_train_epoch_end=False,
                            auto_insert_metric_name=False
                        ),
                        ModelCheckpoint(
                            monitor=f"validation_loss_{test_type}{model_type}/valid_and_unique",
                            filename=f"best_valid_and_unique_{test_type}{model_type}"+"_{epoch:06d}-{"+f"validation_loss_{test_type}{model_type}/valid_and_unique"+":.4f}",
                            save_top_k=1,
                            mode="max",
                            save_on_train_epoch_end=False,
                            auto_insert_metric_name=False
                        ),
                        ModelCheckpoint(
                            monitor=f"validation_loss_{test_type}{model_type}/valid_and_unique_and_novel",
                            filename=f"best_valid_and_unique_and_novel_{test_type}{model_type}"+"_{epoch:06d}-{"+f"validation_loss_{test_type}{model_type}/valid_and_unique_and_novel"+":.4f}",
                            save_top_k=1,
                            mode="max",
                            save_on_train_epoch_end=False,
                            auto_insert_metric_name=False
                        ),
                    ]
                else:
                    callbacks += [
                        ModelCheckpoint(
                            monitor=f"validation_loss_{test_type}{model_type}/spectrum_reconstruction_loss",
                            filename=f"best_spectrum_reconstruction_loss_{test_type}{model_type}"+"_{epoch:06d}-{"+f"validation_loss_{test_type}{model_type}/spectrum_reconstruction_loss"+":.4f}",
                            save_top_k=1,
                            mode="min",
                            save_on_train_epoch_end=False,
                            auto_insert_metric_name=False
                        ),
                        ModelCheckpoint(
                            monitor=f"validation_loss_{test_type}{model_type}/normalized_spectrum_reconstruction_loss",
                            filename=f"best_normalized_spectrum_reconstruction_loss_{test_type}{model_type}"+"_{epoch:06d}-{"+f"validation_loss_{test_type}{model_type}/normalized_spectrum_reconstruction_loss"+":.4f}",
                            save_top_k=1,
                            mode="min",
                            save_on_train_epoch_end=False,
                            auto_insert_metric_name=False
                        ),
                        ModelCheckpoint(
                            monitor=f"validation_loss_{test_type}{model_type}/mean_MMD_ratio_3",
                            filename=f"best_mean_MMD_ratio_3_{test_type}{model_type}"+"_{epoch:06d}-{"+f"validation_loss_{test_type}{model_type}/mean_MMD_ratio_3"+":.4f}",
                            save_top_k=1,
                            mode="min",
                            save_on_train_epoch_end=False,
                            auto_insert_metric_name=False
                        ),
                        ModelCheckpoint(
                            monitor=f"validation_loss_{test_type}{model_type}/mean_MMD_ratio_4",
                            filename=f"best_mean_MMD_ratio_4_{test_type}{model_type}"+"_{epoch:06d}-{"+f"validation_loss_{test_type}{model_type}/mean_MMD_ratio_4"+":.4f}",
                            save_top_k=1,
                            mode="min",
                            save_on_train_epoch_end=False,
                            auto_insert_metric_name=False
                        ),
                        ModelCheckpoint(
                            monitor=f"validation_loss_{test_type}{model_type}/mean_MMD_ratio_5",
                            filename=f"best_mean_MMD_ratio_5_{test_type}{model_type}"+"_{epoch:06d}-{"+f"validation_loss_{test_type}{model_type}/mean_MMD_ratio_5"+":.4f}",
                            save_top_k=1,
                            mode="min",
                            save_on_train_epoch_end=False,
                            auto_insert_metric_name=False
                        ),
                        ModelCheckpoint(
                            monitor=f"validation_loss_{test_type}{model_type}/mmd_spectral",
                            filename=f"best_mmd_spectral_{test_type}{model_type}"+"_{epoch:06d}-{"+f"validation_loss_{test_type}{model_type}/mmd_spectral"+":.4f}",
                            save_top_k=1,
                            mode="min",
                            save_on_train_epoch_end=False,
                            auto_insert_metric_name=False
                        ),
                        ModelCheckpoint(
                            monitor=f"validation_loss_{test_type}{model_type}/mmd_wavelet",
                            filename=f"best_mmd_wavelet_{test_type}{model_type}"+"_{epoch:06d}-{"+f"validation_loss_{test_type}{model_type}/mmd_wavelet"+":.4f}",
                            save_top_k=1,
                            mode="min",
                            save_on_train_epoch_end=False,
                            auto_insert_metric_name=False
                        ),
            ]
    
    if args.log_grads:
        callbacks.append(GradMonitor())

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    print(f"MODEL: {version_string}")
    trainer.fit(model, data_module)

    eval_data_module = GraphDataModule(**data_module_params, eval_MMD=True) # NOTE: to compute correct MMD rations (vs test set) eval_MMD=True flag is needed for the data module
    eval_data_module.setup()
    trainer.test(model, datamodule=eval_data_module)
