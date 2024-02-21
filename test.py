import torch
import yaml
import types
import json

import numpy as np

from data.lightning.CelebAHQDataModule import CelebAHQDataModule
from data.lightning.FFHQDataModule import FFHQDataModule

from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.rcGAN import rcGAN
from models.lightning.EigenGAN import EigenGAN
import matplotlib.pyplot as plt
from matplotlib import gridspec
from utils.embeddings import InceptionEmbedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    fname = 'configs/celebahq.yml'
    if args.ffhq:
        fname = 'configs/celebahq.yml'

    with open(fname, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    if args.ffhq:
        dm = FFHQDataModule(cfg)
    else:
        dm = CelebAHQDataModule(cfg)
    dm.setup()
    test_loader = dm.test_dataloader()
    val_loader = dm.val_dataloader()
    train_loader = dm.train_dataloader()
    inception_embedding = InceptionEmbedding()

    with torch.no_grad():
        if args.eigengan:
            model = EigenGAN.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')
        else:
            model = rcGAN.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')

        model.cuda()
        model.eval()

        cfid_metric = CFIDMetric(gan=model,
                                 loader=test_loader,
                                 image_embedding=inception_embedding,
                                 condition_embedding=inception_embedding,
                                 cuda=True,
                                 args=cfg,
                                 train_loader=False,
                                 num_samps=32)

        cfid_val_1 = cfid_metric.get_cfid_torch_pinv().cpu().numpy()

        cfid_metric = CFIDMetric(gan=model,
                                 loader=test_loader,
                                 image_embedding=inception_embedding,
                                 condition_embedding=inception_embedding,
                                 cuda=True,
                                 args=cfg,
                                 dev_loader=val_loader,
                                 train_loader=False,
                                 num_samps=8)

        cfids = cfid_metric.get_cfid_torch_pinv().cpu().numpy()

        cfid_val_2 = np.mean(cfids)

        cfid_metric = CFIDMetric(gan=model,
                                 loader=test_loader,
                                 image_embedding=inception_embedding,
                                 condition_embedding=inception_embedding,
                                 cuda=True,
                                 args=cfg,
                                 train_loader=train_loader,
                                 dev_loader=val_loader,
                                 num_samps=1)

        cfids = cfid_metric.get_cfid_torch_pinv().cpu().numpy()

        cfid_val_3 = np.mean(cfids)

        print(f'CFID_1: {cfid_val_1}')
        print(f'CFID_2: {cfid_val_2}')
        print(f'CFID_3: {cfid_val_3}')