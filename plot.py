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

def load_object(dct):
    return types.SimpleNamespace(**dct)

# TODO: Sort by LPIPS
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
    fig_count = 1
    dm.setup()
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        rcGAN_model = rcGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/rcgan/checkpoint_best.ckpt')

        rcGAN_model.cuda()

        rcGAN_model.eval()

        eigenGAN_model = EigenGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/eigengan/checkpoint_best.ckpt')

        eigenGAN_model.cuda()

        eigenGAN_model.eval()

        for i, data in enumerate(test_loader):
            y, x, mask, mean, std = data[0]
            y = y.cuda()
            x = x.cuda()
            mask = mask.cuda()
            mean = mean.cuda()
            std = std.cuda()

            gens_rcgan = torch.zeros(
                size=(y.size(0), cfg.num_z_test, 3, cfg.im_size, cfg.im_size)).cuda()
            gens_eigengan = torch.zeros(
                size=(y.size(0), cfg.num_z_test, 3, cfg.im_size, cfg.im_size)).cuda()

            for z in range(cfg.num_z_test):
                gens_rcgan[:, z, :, :, :] = rcGAN_model.forward(y, mask) * std[:, :, None, None] + mean[:, :, None, None]
                gens_eigengan[:, z, :, :, :] = eigenGAN_model.forward(y, mask) * std[:, :, None, None] + mean[:, :, None, None]

            gt = x * std[:, :, None, None] + mean[:, :, None, None]
            zfr = y * std[:, :, None, None] + mean[:, :, None, None]

            for j in range(y.size(0)):
                np_samps = {
                    'rcgan': [],
                    'eigengan': []
                }

                np_gt = None

                np_gt = gt[j].cpu().numpy()
                np_zfr = zfr[j].cpu().numpy()

                for z in range(cfg.num_z_test):
                    np_samps['rcgan'].append(gens_rcgan[j, z].cpu().numpy())
                    np_samps['eigengan'].append(gens_eigengan[j, z].cpu().numpy())


                methods = ['rcgan', 'eigengan']

                # Global recon, error, std
                nrow = len(methods)
                ncol = 7

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                ax = plt.subplot(gs[0, 0])
                ax.imshow(np.transpose(np_gt, (1, 2, 0)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("x")

                ax = plt.subplot(gs[0, 1])
                ax.imshow(np.transpose(np_zfr, (1, 2, 0)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("y")

                for k in range(len(methods)):
                    for l in range(5):
                        ax = plt.subplot(gs[k, l + 2])
                        im = ax.imshow(np.transpose(np_samps[methods[k]][l], (1, 2, 0)))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(f"{methods[k]} {l+1}")

                plt.savefig(f'figures/inpainting/example_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                if fig_count == args.num_figs:
                    exit()
                fig_count += 1
