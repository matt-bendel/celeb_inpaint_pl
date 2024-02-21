import random
import os
import torch

import numpy as np
import matplotlib.pyplot as plt
# TODO: REMOVE
################
from utils.parse_args import create_arg_parser
from data_loaders.prepare_data import create_data_loaders
from wrappers.our_gen_wrapper import load_best_gan
from evaluation_scripts.lpips.lpips_metric import LPIPSMetric

def get_lpips(args, G, test_loader, num_runs, t, truncation_latent=None):
    lpips_metric = LPIPSMetric(G, test_loader)
    LPIPS = lpips_metric.compute_lpips(num_runs, t, truncation_latent)
    print('LPIPS: ', LPIPS)
    return LPIPS

if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    torch.backends.cudnn.benchmark = True

    args = create_arg_parser().parse_args()
    # restrict visible cuda devices
    if args.data_parallel or (args.device >= 0):
        if not args.data_parallel:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    args.in_chans = 3
    args.out_chans = 3

    args.checkpoint_dir = '/storage/matt_models/inpainting/cvpr_ours_256'

    G = load_best_gan(args)
    G.update_gen_status(val=True)

    train_loader, val_loader, test_loader = create_data_loaders(args)


    get_lpips(args, G, test_loader, 1, None, truncation_latent=None)
