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
from evaluation_scripts.lpips.lpips_metric import LPIPSMetric

def load_object(dct):
    return types.SimpleNamespace(**dct)

def get_lpips(args, G, test_loader, num_runs):
    lpips_metric = LPIPSMetric(G, test_loader)
    LPIPS = lpips_metric.compute_lpips(num_runs)
    print('LPIPS: ', LPIPS)
    return LPIPS

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
    fig_count = 1
    dm.setup()
    test_loader = dm.test_dataloader()

    count = 0

    with torch.no_grad():
        method = 'eigengan'
        model = EigenGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')

        model.cuda()
        model.eval()

        get_lpips(args, model, test_loader, 1)


