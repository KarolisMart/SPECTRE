import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from full_gan import GraphDataModule
from full_gan import SPECTRE

# Don't log warning messages
import logging
logging.basicConfig(level=logging.ERROR)

if __name__ == '__main__':
    # EXAMPLE: test.py --checkpoint 'logs/SPECTRE_rot_3_en_0.01_0.005_0.005_beta_0.5_0.9_wd_0.0_k_2_G_d_0.1_D_d_0.1_instance_norm_D_1_n_G_8_D_8_hid_G_64_D_64_SN_GGELU_DGELU_bs_10_lr_G_0.0001_D_0.0001_community_12-21-100_norm_20220120-002951_j36150833/checkpoints/best_mean_MMD_ratio_3_all_fake_ema_002879-0.5006.ckpt' --dataset community --n_start 12 --n_end 21 --n_graphs 100 --compute_emd
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--experiment_name', default='', type=str)
    parser.add_argument('--log_grads', default=False, action="store_true")
    parser.add_argument('--job_id', default='', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--compute_emd', default=False, action="store_true")
    parser = GraphDataModule.add_data_specific_args(parser)
    # parser = SPECTRE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    # Modify trainer defaults
    parser.set_defaults(gpus=1)
    # ------------------------------------------------------------------------
    args = parser.parse_args()

    model = SPECTRE.load_from_checkpoint(args.checkpoint)

    model.hparams.compute_emd = args.compute_emd

    # Follow the model on ignoring the first eigenvalue and eigenvector
    args.ignore_first_eigv = model.hparams.ignore_first_eigv

    checkpoint_folder = os.path.dirname(os.path.dirname(args.checkpoint))
    version_string = os.path.basename(os.path.normpath(checkpoint_folder))
    logger = TensorBoardLogger(save_dir='logs', name=args.experiment_name, version=version_string) #version="Custom experiment version string"
    args.logger = logger
    trainer = pl.Trainer.from_argparse_args(args, resume_from_checkpoint=args.checkpoint) #progress_bar_refresh_rate=0 to not display progress bar on the server
    print(f"MODEL: {args.checkpoint}")

    data_module = GraphDataModule(batch_size=args.batch_size, k=model.hparams.k_eigval, n_nodes=args.n_nodes, n_graphs=args.n_graphs,
                                  n_data_workers=args.n_data_workers, same_sample=args.same_sample, n_start=args.n_start, n_end=args.n_end,
                                  dataset=args.dataset, validate_on_train_cond=args.validate_on_train_cond, ignore_first_eigv=args.ignore_first_eigv,
                                  qm9_strict_eval=args.qm9_strict_eval, eval_MMD=True, compute_emd=args.compute_emd)

    # Not needed if we test on the same dataset as we train on. Uncomment if dataset/n_max changes.
    # data_module.setup()
    # model.n_max = data_module.n_max
    # print(f'Largest graph has {data_module.n_max} nodes')

    trainer.test(model, datamodule=data_module)
