import os
import argparse

home_path = os.getenv("HOME")

parser = argparse.ArgumentParser(description='TSFNet')

# Init
parser.add_argument('--model', type=str, default="TSFNet",
                    help='TSFNet')
parser.add_argument("--pre_train", type=int, default=0, 
                    help="load pretrain model or not?")
parser.add_argument("--start_epoch", type=int, default=0,
                    help="Start epoch while training.")
parser.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of input color channels to use')
parser.add_argument('--o_colors', type=int, default=3,
                    help='number of output color channels to use')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function[relu, prelu, leakyrelu, gelu]')
parser.add_argument('--deformable', type=bool, default=True,
                    help="Use deformable convolution in the convolution")

# MST
parser.add_argument('--downsample_1', type=int, default=4,
                    help='downsample for first input')
parser.add_argument('--downsample_2', type=int, default=16,
                    help='downsample for second input')
parser.add_argument('--n_feats_1', type=int, default=64,
                    help='number of feature maps fro first input')
parser.add_argument('--n_feats_2', type=int, default=64,
                    help='number of feature maps for second input')

# parameters
parser.add_argument('--save_every', type=int, default=1,
                    help='save model per every N epoch')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
args = parser.parse_args()