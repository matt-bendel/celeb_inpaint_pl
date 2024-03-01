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
        rcGAN_model = rcGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/rcgan/checkpoint_best.ckpt')

        rcGAN_model.cuda()

        rcGAN_model.eval()

        eigenGAN_model = EigenGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/eigengan/checkpoint_best.ckpt')

        eigenGAN_model.cuda()

        eigenGAN_model.eval()

        coModGAN_model = CoModGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/comodgan/checkpoint_best.ckpt')

        coModGAN_model.cuda()

        coModGAN_model.eval()

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
            gens_comodgan = torch.zeros(
                size=(y.size(0), cfg.num_z_test, 3, cfg.im_size, cfg.im_size)).cuda()

            for z in range(cfg.num_z_test):
                gens_rcgan[:, z, :, :, :] = rcGAN_model.forward(y, mask) * std[:, :, None, None] + mean[:, :, None, None]
                gens_eigengan[:, z, :, :, :] = eigenGAN_model.forward(y, mask) * std[:, :, None, None] + mean[:, :, None, None]
                gens_comodgan[:, z, :, :, :] = coModGAN_model.forward(y, mask) * std[:, :, None, None] + mean[:, :, None, None]

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

                for z in range(cfg.num_z_test):
                    np_samps['comodgan'].append(gens_comodgan[j, z].cpu().numpy())
                    np_samps['rcgan'].append(gens_rcgan[j, z].cpu().numpy())
                    np_samps['eigengan'].append(gens_eigengan[j, z].cpu().numpy())

                cov_mat_rcgan = np.zeros((20, 3 * np_gt.shape[-1] * np_gt.shape[-2]))
                cov_mat_eigengan = np.zeros((20, 3 * np_gt.shape[-1] * np_gt.shape[-2]))
                cov_mat_comodgan = np.zeros((20, 3 * np_gt.shape[-1] * np_gt.shape[-2]))

                for z in range(20):
                    cov_mat_rcgan[z, :] = np_samps['rcgan'][z].flatten()
                    cov_mat_eigengan[z, :] = np_samps['eigengan'][z].flatten()
                    cov_mat_comodgan[z, :] = np_samps['comodgan'][z].flatten()

                _, _, vh_rcgan = np.linalg.svd(cov_mat_rcgan, full_matrices=False)
                _, _, vh_eigengan = np.linalg.svd(cov_mat_eigengan, full_matrices=False)
                _, _, vh_comodgan = np.linalg.svd(cov_mat_comodgan, full_matrices=False)

                np_vh = {
                    'comodgan': vh_comodgan,
                    'rcgan': vh_rcgan,
                    'eigengan': vh_eigengan
                }
                methods = ['comodgan', 'rcgan', 'eigengan']

                # Global recon, error, std
                nrow = len(methods)
                ncol = 2

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))


                for k in range(len(methods)):
                    for l in range(2):
                        ax = plt.subplot(gs[k, l + 2])
                        im_np = np_vh[methods[k]][l].reshape((3, 256, 256))
                        im_np = (im_np - np.min(im_np)) / (np.max(im_np) - np.min(im_np))
                        im = ax.imshow(im_np.transpose(1, 2, 0))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])

                plt.savefig(f'figures/inpainting_evecs/example_{fig_count}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                if fig_count == args.num_figs:
                    exit()
                fig_count += 1
