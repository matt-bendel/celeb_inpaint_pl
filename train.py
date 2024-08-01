import torch
import os
import yaml
import types
import json

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from data.lightning.CelebAHQDataModule import CelebAHQDataModule
from data.lightning.FFHQDataModule import FFHQDataModule
from utils.parse_args import create_arg_parser
from models.lightning.rcGAN import rcGAN
from models.lightning.EigenGAN import EigenGAN
from models.lightning.CoModGAN import CoModGAN
from models.lightning.Ohayon import Ohayon

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")
    print(f"Number of GPUs: {args.num_gpus}")

    if args.inpaint:
        fname = 'configs/celebahq.yml'
        if args.ffhq:
            fname = 'configs/ffhq.yml'

        with open(fname, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        # if args.eigengan:
            # cfg.batch_size = 2

        if args.ffhq:
            dm = FFHQDataModule(cfg)
        else:
            dm = CelebAHQDataModule(cfg)

        if args.eigengan:
            model = EigenGAN(cfg, args.exp_name, args.num_gpus)
        elif args.comodgan:
            model = CoModGAN(cfg, args.exp_name, args.num_gpus)
        elif args.ohayon:
            model = Ohayon(cfg, args.exp_name, args.num_gpus)
        else:
            model = rcGAN(cfg, args.exp_name, args.num_gpus)

    else:
        print("No valid application selected. Please include one of the following args: --mri")
        exit()

    wandb_logger = WandbLogger(
        project="neurips",
        name=args.exp_name,
        log_model="all",
        save_dir=cfg.checkpoint_dir + 'wandb'
    )

    checkpoint_callback_epoch = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=cfg.checkpoint_dir + args.exp_name + '/',
        filename='checkpoint-{epoch}',
        save_top_k=50
    )

    trainer = pl.Trainer(accelerator="gpu", devices=args.num_gpus, strategy='ddp',
                         max_epochs=cfg.num_epochs, callbacks=[checkpoint_callback_epoch],
                         num_sanity_val_steps=2, profiler="simple", logger=wandb_logger, benchmark=False,
                         log_every_n_steps=10)

    if args.resume:
        trainer.fit(model, dm,
                    ckpt_path=cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={args.resume_epoch}.ckpt')
    else:
        trainer.fit(model, dm)
