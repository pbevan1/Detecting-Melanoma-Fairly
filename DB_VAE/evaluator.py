import os
from tqdm import tqdm
from typing import Optional, List
import torch
from torch.utils.data import DataLoader
from DB_VAE.vae_model import Db_vae
import DB_VAE.utils
from DB_VAE.logger import logger
from DB_VAE.dataset import *


class Evaluator:
    """
    Class that evaluates a model based on a given pre-initialized model or path_to_model
    and displays several performance metrics.
    """
    def __init__(
        self,
        args,
        device: str,
        model: Optional[Db_vae] = None,
        **kwargs
    ):
        self.args = args
        self.z_dim = args.z_dim
        self.device = device
        self.batch_size = args.batch_size
        self.model_name = args.model_name

        self.path_to_model = args.test_no
        self.model: Db_vae = self.init_model(self.path_to_model, model)
        _, df_val, _, _, _, _, _, _, _, _, _, _, _ = get_df()
        self.csv = df_val

    def init_model(self, path_to_model: Optional[str] = None, model: Optional[Db_vae] = None):
        """Initializes a stored model or one that directly comes from training."""
        if model is not None:
            logger.info("Using model passed")
            return model.to(self.device)

        # If path_to_model, load model from file
        if path_to_model:
            return Db_vae.init(path_to_model=path_to_model, device=self.device,
                               z_dim=self.z_dim).to(self.device)

        logger.error(
            "No model or path_to_model given",
            next_step="Evaluation will not run",
            tip="Instantiate with a trained model, or set `path_to_model`."
        )
        raise Exception

    def eval(self, filter_skin_color=None):
        """Evaluates a model based and returns the amount of correctly classified and total classified images."""
        self.model.eval()

        eval_loader: DataLoader = make_eval_loader(
            num_workers=args.num_workers,
            csv=self.csv,
            filter_skin_color=filter_skin_color,
        )

        correct_count, count, LABELS, SIGOUTS = self.eval_model(eval_loader)
        return correct_count, count, LABELS, SIGOUTS

    def eval_on_setups(self, eval_name: Optional[str] = None):
        """Evaluates a model and writes the results to a given file name."""
        eval_name = self.args.eval_name if eval_name is None else eval_name

        if args.dataset == 'Fitzpatrick17k':
            # Define the predefined setups
            skin_list = [0, 1, 2, 3, 4, 5]
            name_list = ["type_1", "type_2", "type_3", "type_4", "type 5", "type 6"]

            # Init the metrics
            accuracies = []
            aucs = []
            correct = 0
            total_count = 0

            # Go through the predefined setup
            for i in range(6):
                logger.info(f"Running setup for {name_list[i]}")

                # Calculate on the current setup
                correct_count, count, LABELS, SIGOUTS = self.eval(
                    filter_skin_color=skin_list[i]
                )

                # Calculate the metrics
                a_u_c = utils.calculate_AUC(LABELS, SIGOUTS)
                accuracy = correct_count / count * 100
                correct += correct_count
                total_count += count

                # Log the recall
                logger.info(f"Accuracy for {name_list[i]} is {accuracy:.3f}")
                logger.info(f"AUC for {name_list[i]} is {a_u_c:.3f}")
                accuracies.append(accuracy)
                aucs.append(a_u_c)

            # Calculate the average recall
            avg_acc = correct/total_count*100
            avg_auc = sum(aucs)/len(name_list)

            acc_variance = (torch.tensor(accuracies)).var().item()
            auc_variance = (torch.tensor(aucs)).var().item()

            # Logger info
            logger.info(f"Accuracy => all: {avg_acc:.3f}")
            logger.info(f"Accuracy => type 1: {accuracies[0]:.3f}")
            logger.info(f"Accuracy => type 2: {accuracies[1]:.3f}")
            logger.info(f"Accuracy => type 3: {accuracies[2]:.3f}")
            logger.info(f"Accuracy => type 4: {accuracies[3]:.3f}")
            logger.info(f"Accuracy => type 5: {accuracies[4]:.3f}")
            logger.info(f"Accuracy => type 6: {accuracies[5]:.3f}")
            logger.info(f"Accuracy Variance => {acc_variance:.3f}")
            logger.info(f"AUC => all: {avg_auc:.3f}")
            logger.info(f"AUC => type 1: {aucs[0]:.3f}")
            logger.info(f"AUC => type 2: {aucs[1]:.3f}")
            logger.info(f"AUC => type 3: {aucs[2]:.3f}")
            logger.info(f"AUC => type 4: {aucs[3]:.3f}")
            logger.info(f"AUC => type 5: {aucs[4]:.3f}")
            logger.info(f"AUC => type 6: {aucs[5]:.3f}")
            logger.info(f"AUC Variance => {auc_variance:.3f}")

            # Write final results
            path_to_eval_results = f"results/logs/{self.args.test_no}/results.txt"
            with open(path_to_eval_results, 'a+') as write_file:

                # If file has no header
                if not os.path.exists(path_to_eval_results) or os.path.getsize(path_to_eval_results) == 0:
                    write_file.write(f"metric,name,type_1,type_2,type_3,type_4,type_5,type_6,avg,var\n")
                # write_file.write(f"{self.path_to_model}_{self.model_name}")
                write_file.write(f"accuracy,{self.path_to_model}_{self.model_name},{accuracies[0]:.3f},"
                                 f"{accuracies[1]:.3f},{accuracies[2]:.3f},{accuracies[3]:.3f},"
                                 f"{accuracies[4]:.3f},{accuracies[5]:.3f},{avg_acc:.3f},{acc_variance:.3f}\n"
                                 f"AUC,{self.path_to_model}_{self.model_name},{aucs[0]:.3f},{aucs[1]:.3f},{aucs[2]:.3f},"
                                 f"{aucs[3]:.3f},{aucs[4]:.3f},{aucs[5]:.3f},"
                                 f"{avg_auc:.3f}{auc_variance:.3f}\n")

        else:
            # Calculate on the current setup
            correct_count, count, LABELS, SIGOUTS = self.eval()

            # Calculate the metrics
            a_u_c = utils.calculate_AUC(LABELS, SIGOUTS)
            accuracy = correct_count / count * 100

            # Log the recall
            logger.info(f"Accuracy on hold-out set is {accuracy:.3f}")
            logger.info(f"AUC on hold-out set is {a_u_c:.3f}")

            # Write final results
            path_to_eval_results = f"results/logs/{self.args.test_no}/results.txt"
            with open(path_to_eval_results, 'a+') as write_file:
                write_file.write(f"{self.path_to_model}_{self.model_name}")
                write_file.write(f"Accuracy: {accuracy}, AUC: {a_u_c}")

        logger.success("Finished evaluation!")

    def eval_model(self, eval_loader: DataLoader):
        """Perform evaluation of a single epoch."""
        self.model.eval()

        count = 0
        correct_count = 0
        SIGOUTS = []
        LABELS = []

        bar = tqdm(eval_loader)
        # Iterate over all images
        for _, batch in enumerate(bar):
            count += 1
            data, labels, _ = batch

            for images, target in zip(data, labels):
                if len(images.shape) == 5:
                    images = images.squeeze(dim=0)

                images = images.unsqueeze(dim=0)

                images = images.to(self.device)

                pred, sigout = self.model.forward_eval(images)

                pred = pred.detach().cpu()

                if (pred > 0 and target == 1) | (pred < 0 and target == 0):
                    correct_count += 1
                    break

                SIGOUTS.append(sigout.detach().cpu())
                LABELS.append(target)

        LABELS = torch.stack(LABELS).numpy()
        SIGOUTS = torch.cat(SIGOUTS).numpy()

        logger.info(f"Amount of labels:{count}, Correct labels:{correct_count}")

        return correct_count, count, LABELS, SIGOUTS
