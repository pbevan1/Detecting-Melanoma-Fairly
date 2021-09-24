import argparse


# See help for information on how to use arguments to run experiments
def parse_args():
    parser = argparse.ArgumentParser()
    # Setting model trunk architecture
    parser.add_argument('--arch', type=str, help='choose from resnext101,'
                                                 ' enet, resnet101, densenet or inception', default='resnext101')
    # Setting training & test dataset
    parser.add_argument('--dataset', type=str, help='choose from ISIC or Fitzpatrick17k', default='ISIC')
    parser.add_argument('--split-skin-types', action='store_true')
    parser.add_argument('--tune', help='Use for tuning, tests only val set', action='store_true')
    # Setting debiasing technique
    parser.add_argument('--debias-config', type=str, help='choose from baseline, LNTL, TABE, both,'
                                                          ' doubleTABE or doubleLNTL', default='baseline')
    parser.add_argument('--GRL', help='Use to add gradient reversal', action='store_true')
    parser.add_argument('--switch-heads', help='switches aux head labels if using 2', action='store_true')
    parser.add_argument('--deep-aux', help='adds fully connected layer to aux', action='store_true')
    # Bias to remove
    parser.add_argument('--sktone', help='uses skin type labels for aux head', action='store_true')
    # Setting hyperparameters
    parser.add_argument('--seed', help='sets all random seeds', type=int, default=0)
    parser.add_argument('--batch-size', help='sets batch size', type=int, default=64)
    parser.add_argument('--num-workers', help='sets number of cpu workers', type=int, default=16)
    parser.add_argument('--lr-base', help='sets baseline learning rate', type=float, default=0.0003)
    parser.add_argument('--lr-class', help='sets TABE learning rate', type=float, default=0.003)
    parser.add_argument('--alpha', help='sets alpha for TABE', type=float, default=0.03)
    parser.add_argument('--lambdaa', help='sets lambda for LNTL', type=float, default=0.01)
    parser.add_argument('--momentum', help='sets momentum', type=float, default=0.9)
    parser.add_argument('--n-epochs', help='sets epochs to train for', type=int, default=15)
    parser.add_argument('--save-epoch', help='sets model save interval', type=int, default=1)
    parser.add_argument('--out-dim', help='sets main head output dimension', type=int, default=1)
    parser.add_argument('--num-aux', help='sets aux head output dimentsion', type=int, default=2)
    parser.add_argument('--num-aux2', help='sets 2nd aux head output dimension', type=int, default=2)
    # Setting directories
    parser.add_argument('--image-dir', help='path to image directory', type=str, default='./data/images')
    parser.add_argument('--csv-dir', help='path to csv directory', type=str, default='./data/csv')
    parser.add_argument('--model-dir', help='path to save models to', type=str, default='./results/weights')
    parser.add_argument('--plot-dir', help='path to save plots to', type=str, default='./results/plots')
    parser.add_argument('--log-dir', help='path to save logs to', type=str, default='./results/logs')
    # Miscellaneous
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--enet-type', help='select type of EfficientNet', type=str, default='efficientnet_b3')
    parser.add_argument('--DEBUG', help='runs for 3 batches per epoch', action='store_true')
    parser.add_argument('--test-only', help='only testing, must have trained weights', action='store_true')
    parser.add_argument('--test-no', help='test number', type=int, required=True)
    parser.add_argument('--cv', help='use to run cross-validation', action='store_true')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', help='selecting GPUs to run on', type=str, default='0')

    args, _ = parser.parse_known_args()
    return args
