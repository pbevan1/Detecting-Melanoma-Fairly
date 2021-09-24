from arguments import parse_args
import os
import torch

'''
This file is used import pseudo global variables into other modules
'''

args = parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

device = torch.device('cuda')
