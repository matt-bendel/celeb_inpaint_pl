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
from models.lightning.CoModGAN import CoModGAN
import matplotlib.pyplot as plt
from matplotlib import gridspec
from utils.embeddings import InceptionEmbedding
from evaluation_scripts.cfid.cfid_metric_dps import CFIDMetric
from evaluation_scripts.fid.fid_metric_dps import FIDMetric


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    fname = 'configs/celebahq.yml'
    if args.ffhq:
        fname = 'configs/ffhq.yml'

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
        fid_metric = FIDMetric(loader=test_loader,
                               image_embedding=inception_embedding,
                               condition_embedding=inception_embedding,
                               cuda=True,
                               args=cfg,
                               dev_loader=val_loader,
                               ref_loader=train_loader)

        fid_val = fid_metric.get_fid()
        print(fid_val)
        exit()

        cfid_metric = CFIDMetric(loader=test_loader,
                                 image_embedding=inception_embedding,
                                 condition_embedding=inception_embedding,
                                 cuda=True,
                                 args=cfg,
                                 train_loader=train_loader,
                                 dev_loader=val_loader,
                                 num_samps=1)

        cfids = cfid_metric.get_cfid_torch_pinv().cpu().numpy()

        cfid_val_3 = np.mean(cfids)

        print(f'CFID_3: {cfid_val_3}')
