import pathlib

from utils.args import Args


def create_arg_parser():
    # CREATE THE PARSER
    parser = Args()

    parser.add_argument('--resume', action='store_true',
                        help='Whether or not to resume training.')
    parser.add_argument('--resume-epoch', default=0, type=int, help='Epoch to resume training from')
    parser.add_argument('--inpaint', action='store_true',
                        help='If the application is inpainting')
    parser.add_argument('--ffhq', action='store_true',
                        help='If the dataset is ffhq')
    parser.add_argument('--eigengan', action='store_true',
                        help='If the model is EigenGAN')
    parser.add_argument('--comodgan', action='store_true',
                        help='If the model is CoModGAN')
    parser.add_argument('--exp-name', type=str, default="", help='Name for the run.')
    parser.add_argument('--num-gpus', default=1, type=int, help='The number of GPUs to use during training.')
    parser.add_argument('--num-figs', default=1, type=int, help='The number of figures to generate while plotting.')

    return parser
