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
    diff_models = ['ddrm', 'ddnm', 'dps']
    diff_models = ['dps']

    fname = 'configs/celebahq.yml'
    if args.ffhq:
        fname = 'configs/ffhq.yml'

    # img_inds = [5907, 9350, 1816, 4372, 11835, 1079, 15312, 14879, 8206, 4940, 17884, 14344, 1965, 3722, 14086, 18843, 14547, 5340, 10731, 11841, 15439, 17479, 5606, 1538, 11212, 13777, 5048, 4303, 246, 5932]
    img_inds = [765, 241, 454, 828, 477, 52, 503, 47, 623, 729, 42, 345, 614, 54, 489, 349, 287, 833, 342, 73, 633, 873, 387, 350, 109, 805, 376, 404, 590, 672]
    img_inds = [765, 241, 454, 828, 477, 109, 287, 342, 387, 623, 633, 729, 805, 503]

    with open(fname, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    if args.ffhq:
        dm = FFHQDataModule(cfg)
    else:
        dm = CelebAHQDataModule(cfg)
    dm.setup()
    test_loader = dm.test_dataloader()

    for diff_model in diff_models:
        fig_count = 1
        count = 0
        running_count = 0

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                y, x, mask, mean, std = data[0]
                y = y.cuda()
                x = x.cuda()
                mean = mean.cuda()
                std = std.cuda()

                if i not in img_inds:
                    running_count += 1
                    count += 1
                    continue

                running_count += 1

                print(i)
                gens = torch.zeros(size=(y.size(0), 5, 3, cfg.im_size, cfg.im_size))
                mask = torch.zeros(mask.shape)
                for j in range(x.shape[0]):
                    for k in [1,6,7,8,9]:
                        gens[j, k] = torch.load(f'/storage/matt_models/inpainting/{diff_model}/test_20k/image_{count + j}_sample_{k}.pt') * std[j, :, None, None].cpu() + mean[j, :, None, None].cpu()
                    mask[j] = torch.load(f'/storage/matt_models/inpainting/dps/test/image_{count + j}_mask.pt')

                gens = gens.cuda()
                mask = mask.cuda()

                count += y.shape[0]

                y = x * mask

                gt = x * std[:, :, None, None] + mean[:, :, None, None]
                zfr = y * std[:, :, None, None] + mean[:, :, None, None]

                gens_dc = torch.zeros(gens.shape).cuda()
                for k in range(5):
                    gens_dc[:, k, :, :, :] = gens[:, k, :, :, :] * (1 - mask) + gt * mask

                for j in range(y.size(0)):
                    np_gt = gt[j].cpu().numpy()
                    np_zfr = zfr[j].cpu().numpy()

                    np_samps = gens[j].cpu().numpy()
                    np_samps_dc = gens_dc[j].cpu().numpy()

                    # Global recon, error, std
                    nrow = 1
                    ncol = 1

                    # fig = plt.figure(figsize=(ncol + 1, nrow + 1))
                    #
                    # gs = gridspec.GridSpec(nrow, ncol,
                    #                        wspace=0.0, hspace=0.0,
                    #                        top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                    #                        left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
                    #
                    # ax = plt.subplot(gs[0, 0])
                    # im = ax.imshow(np.transpose(np_gt, (1, 2, 0)))
                    # ax.set_xticklabels([])
                    # ax.set_yticklabels([])
                    # ax.set_xticks([])
                    # ax.set_yticks([])
                    #
                    # plt.savefig(f'figures/rebuttal/original_{diff_model}_{running_count - 1}.png', bbox_inches='tight', dpi=300)
                    # plt.close(fig)

                    nrow = 1
                    ncol = 1

                    # fig = plt.figure(figsize=(ncol + 1, nrow + 1))
                    #
                    # gs = gridspec.GridSpec(nrow, ncol,
                    #                        wspace=0.0, hspace=0.0,
                    #                        top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                    #                        left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
                    #
                    # ax = plt.subplot(gs[0, 0])
                    # im = ax.imshow(np.transpose(np_zfr, (1, 2, 0)))
                    # ax.set_xticklabels([])
                    # ax.set_yticklabels([])
                    # ax.set_xticks([])
                    # ax.set_yticks([])
                    #
                    # plt.savefig(f'figures/rebuttal/masked_{diff_model}_{fig_count}.png', bbox_inches='tight', dpi=300)
                    # plt.close(fig)

                    nrow = 1
                    ncol = 5

                    fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                    gs = gridspec.GridSpec(nrow, ncol,
                                           wspace=0.0, hspace=0.0,
                                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                    for l in range(5):
                        ax = plt.subplot(gs[0, l])
                        im = ax.imshow(np.transpose(np_samps[l], (1, 2, 0)))
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # ax = plt.subplot(gs[0, 1])
                    # im = ax.imshow(np.transpose(np_samps_dc[0], (1, 2, 0)))
                    # ax.set_xticklabels([])
                    # ax.set_yticklabels([])
                    # ax.set_xticks([])
                    # ax.set_yticks([])
                        # ax.set_title(f"{methods[k]} {l+1}")

                    plt.savefig(f'figures/rebuttal/samps_{diff_model}_{running_count-1}.png', bbox_inches='tight', dpi=300)
                    plt.close(fig)

                    # fig = plt.figure(figsize=(ncol + 1, nrow + 1))
                    #
                    # gs = gridspec.GridSpec(nrow, ncol,
                    #                        wspace=0.0, hspace=0.0,
                    #                        top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                    #                        left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
                    #
                    # ax = plt.subplot(gs[0, 0])
                    # im = ax.imshow(np.transpose(np_samps_dc[0], (1, 2, 0)))
                    # ax.set_xticklabels([])
                    # ax.set_yticklabels([])
                    # ax.set_xticks([])
                    # ax.set_yticks([])
                    #
                    # plt.savefig(f'figures/rebuttal/samps_{diff_model}_dc_{fig_count}.png', bbox_inches='tight', dpi=300)
                    # plt.close(fig)

                    fig_count += 1

