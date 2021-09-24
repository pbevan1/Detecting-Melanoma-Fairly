import torch
import os
import argparse
from DB_VAE.logger import logger


# Default device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse arguments
parser = argparse.ArgumentParser()
# Setting training & test dataset
parser.add_argument('--dataset', type=str, help='choose from "ISIC" or "Fitzpatrick17k"', default='ISIC')
parser.add_argument('--split-skin-types', action='store_true')

parser.add_argument('--DEBUG', action='store_true')
parser.add_argument('--DP', action='store_true')
parser.add_argument('--seed', type=int, help='random seed', default=0)
parser.add_argument('--load-model', action='store_true')
# parser.add_argument('--fitzpatrick17k', action='store_true')
parser.add_argument('--image-dir', type=str, default='./data/images/')
parser.add_argument('--csv-dir', type=str, default='./data/csv')
parser.add_argument('--run-folder', type=str, default='outputs')
parser.add_argument('--test-no', type=int, default=0)
parser.add_argument('--image-size', type=int, help='size of images', default=256)
parser.add_argument('--batch-size', type=int, help='size of batch', default=64)
parser.add_argument('--num-workers', type=int, help='number of workers', default=16)
parser.add_argument('--epochs', type=int, help='max number of epochs', default=100)
# parser.add_argument('--save-epoch', type=int, help='epoch to save on', default=100)
parser.add_argument('--z-dim', type=int, help='dimensionality of latent space', default=512)
parser.add_argument('--alpha', type=float, help='importance of debiasing', default=0.01)
parser.add_argument('--num-bins', type=int, help='number of bins', default=10)
parser.add_argument('--debias-type', type=str, help='type of debiasing used', default='none')
# parser.add_argument("--folder_name", type=str, help='folder_name_to_save in')
parser.add_argument("--eval-name", type=str, help='eval name', default='evaluation_results.txt')
parser.add_argument('--model-name', type=str, help='name of the model to evaluate', default='model.pth')
parser.add_argument('--hist-size', type=bool, help='Number of histogram', default=1000)
parser.add_argument('--run-mode', type=str, help="Choose from 'train', 'eval', 'both', 'perturb',"
                                                 " 'interpolate'", default='both')
parser.add_argument('--interp1', type=int, help='first image to interpolate', default=71)
parser.add_argument('--interp2', type=int, help='second image to interpolate', default=6)
parser.add_argument('--var-to-perturb', type=int, help='number of latent variables to perturb', default=50)
parser.add_argument('--perturb-single', action='store_true')
parser.add_argument('-f', type=str, help='Path to kernel json')


class EmptyObject():
    def __getattribute__(self, idx):
        return None


args, unknown = parser.parse_known_args()
if len(unknown) > 0:
    logger.warning(f'There are some unknown args: {unknown}')


def create_folder_name(foldername):
    if foldername == "":
        return foldername

    suffix = ''
    count = 0
    while True:
        if not os.path.isdir(f"results/{foldername}{suffix}"):
            foldername = f'{foldername}{suffix}'
            return foldername
        else:
            count += 1
            suffix = f'_{count}'


def init_trainining_results():
    # Write run-folder name
    if not os.path.exists("results"):
        os.makedirs("results")

    logger.save(f"Saving new run files to {args.test_no}")
    os.makedirs(f'results/plots/{args.test_no}/best_and_worst', exist_ok=True)
    os.makedirs(f'results/plots/{args.test_no}/bias_probs', exist_ok=True)
    os.makedirs(f'results/plots/{args.test_no}/reconstructions/perturbations', exist_ok=True)
    os.makedirs(f'results/logs/{args.test_no}', exist_ok=True)
    os.makedirs(f'results/weights/{args.test_no}', exist_ok=True)

    with open(f"results/logs/{args.test_no}/flags.txt", "w") as write_file:
        write_file.write(f"z_dim = {args.z_dim}\n")
        write_file.write(f"alpha = {args.alpha}\n")
        write_file.write(f"epochs = {args.epochs}\n")
        write_file.write(f"batch size = {args.batch_size}\n")
        write_file.write(f"debiasing type = {args.debias_type}\n")

    with open(f"results/logs/{args.test_no}/training_results.csv", "a+") as write_file:
        write_file.write("epoch,train_loss,valid_loss,train_acc,valid_acc,loss_recon\n")

    with open(f"results/logs/{args.test_no}/flags.txt", "w") as wf:
        wf.write(f"debias_type: {args.debias_type}\n")
        wf.write(f"alpha: {args.alpha}\n")
        wf.write(f"z_dim: {args.z_dim}\n")
        wf.write(f"batch_size: {args.batch_size}\n")
