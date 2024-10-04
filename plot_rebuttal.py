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

    count = 0

    img_inds = [5907, 9350, 1816, 4372, 11835, 1079, 15312, 14879, 8206, 4940, 17884, 14344, 1965, 3722, 14086, 18843, 14547, 5340, 10731, 11841, 15439, 17479, 5606, 1538, 11212, 13777, 5048, 4303, 246, 5932]

    with torch.no_grad():
        method = 'eigengan'
        model = EigenGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')
        model.cuda()
        model.eval()
        running_count = 0

        for i, data in enumerate(test_loader):
            y, x, mask, mean, std = data[0]
            y = y.cuda()
            x = x.cuda()
            mean = mean.cuda()
            std = std.cuda()

            mask = torch.zeros(mask.shape)
            for j in range(x.shape[0]):
                mask[j] = torch.load(f'/storage/matt_models/inpainting/dps/test_20k/image_{count + j}_mask.pt')

            mask = mask.cuda()

            count += y.shape[0]

            y = x * mask

            # if i < 10:
            #     continue

            gens = torch.zeros(
                size=(y.size(0), cfg.num_z_test, 3, cfg.im_size, cfg.im_size)).cuda()

            for z in range(cfg.num_z_test):
                gens[:, z, :, :, :] = model.forward(y, mask) * std[:, :, None, None] + mean[:, :, None, None]

            gt = x * std[:, :, None, None] + mean[:, :, None, None]
            zfr = y * std[:, :, None, None] + mean[:, :, None, None]

            for j in range(y.size(0)):
                print(running_count)
                if running_count not in img_inds:
                    running_count += 1
                    continue

                if running_count not in img_inds:
                    running_count += 1
                    continue

                running_count += 1

                np_gt = gt[j].cpu().numpy()
                np_zfr = zfr[j].cpu().numpy()

                np_samps = gens[j].cpu().numpy()

                # Global recon, error, std
                nrow = 1
                ncol = 5

                fig = plt.figure(figsize=(ncol + 1, nrow + 1))

                gs = gridspec.GridSpec(nrow, ncol,
                                       wspace=0.0, hspace=0.0,
                                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

                for l in range(5):
                    ax = plt.subplot(gs[0, 0])
                    im = ax.imshow(np.transpose(np_samps[l], (1, 2, 0)))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])


                plt.savefig(f'figures/rebuttal/samps_pcagan_{fig_count}_{l}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                fig_count += 1

            if fig_count >= 11:
                exit()