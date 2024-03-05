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

def load_object(dct):
    return types.SimpleNamespace(**dct)

# TODO: Sort by LPIPS
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

    with torch.no_grad():
        if args.eigengan:
            method = 'eigengan'
            model = EigenGAN.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')
        elif args.comodgan:
            method = 'comodgan'
            model = CoModGAN.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')
        else:
            method = 'rcgan'
            model = rcGAN.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')
        model.cuda()
        model.eval()

        for i, data in enumerate(test_loader):
            y, x, mask, mean, std = data[0]
            y = y.cuda()
            x = x.cuda()
            mask = mask.cuda()
            mean = mean.cuda()
            std = std.cuda()

            gens = torch.zeros(
                size=(y.size(0), 50, 3, cfg.im_size, cfg.im_size)).cuda()

            for z in range(50):
                gens[:, z, :, :, :] = model.forward(y, mask) * std[:, :, None, None] + mean[:, :, None, None]

            gt = x * std[:, :, None, None] + mean[:, :, None, None]
            zfr = y * std[:, :, None, None] + mean[:, :, None, None]

            for j in range(y.size(0)):
                np_samps = {
                    'comodgan': [],
                    'rcgan': [],
                    'eigengan': []
                }

                np_gt = None

                np_gt = gt[j].cpu().numpy()
                np_zfr = zfr[j].cpu().numpy()

                # TODO: Get evecs

                for z in range(50):
                    np_samps = gens[j].cpu().numpy()

                cov_mat = np.zeros((50, 3 * np_gt.shape[-1] * np_gt.shape[-2]))
                samp_avg = torch.mean(gens[j], dim=0).cpu().numpy()

                for z in range(50):
                    cov_mat[z, :] = np_samps[z].flatten() - samp_avg.flatten()

                _, _, vh = np.linalg.svd(cov_mat, full_matrices=False)

                nrow = 1
                ncol = 5

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))


                for l in range(5):
                    ax = plt.subplot(gs[0, l])
                    im_np = vh[l].reshape((3, 256, 256))
                    im_np = (im_np - np.min(im_np)) / (np.max(im_np) - np.min(im_np))
                    im = ax.imshow(im_np.transpose(1, 2, 0))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

                plt.savefig(f'figures/inpainting_evecs/example_{method}_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                if fig_count == args.num_figs:
                    exit()
                fig_count += 1
