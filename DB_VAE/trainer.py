import torch
from typing import Optional
import datetime
import pickle
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data.sampler import RandomSampler
from torchvision.utils import make_grid
from DB_VAE.vae_model import Db_vae
import DB_VAE.utils
from DB_VAE.dataset import *
from DB_VAE.generic import *


class Trainer:
    def __init__(
        self,
        args,
        device,
        lr: float = 0.001,
        optimizer=torch.optim.Adam,
        **kwargs
    ):
        """Wrapper class which trains a model."""
        init_trainining_results()
        self.args = args
        self.epochs = args.epochs
        self.load_model = args.load_model
        self.z_dim = args.z_dim
        self.path_to_model = args.test_no
        self.batch_size = args.batch_size
        self.hist_size = args.hist_size
        self.alpha = args.alpha
        self.num_bins = args.num_bins
        self.debias_type = args.debias_type
        self.device = device

        new_model: Db_vae = Db_vae(
            args=self.args,
            z_dim=self.z_dim,
            hist_size=self.hist_size,
            alpha=self.alpha,
            num_bins=self.num_bins,
            device=self.device,
        ).to(device=self.device)

        self.model = self.init_model()

        self.optimizer = optimizer(params=self.model.parameters(), lr=lr)

        df_train, df_val, df_test_atlasD, df_test_atlasC, df_test_ASAN, \
        df_test_MClassD, df_test_MClassC, df_34, df_56, mel_idx = get_df()

        if self.args.DEBUG:
            df_train = df_train.sample(self.batch_size * 3)

        self.df_train = df_train
        self.train_len = len(df_train)
        self.df_valid = df_val

        dataset_train = GenericImageDataset(csv=self.df_train)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size,
                                                   sampler=RandomSampler(dataset_train),
                                                   num_workers=self.args.num_workers, drop_last=True)

        dataset_valid = GenericImageDataset(csv=self.df_valid)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=self.batch_size,
                                                   sampler=RandomSampler(dataset_valid),
                                                   num_workers=self.args.num_workers, drop_last=True)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def init_model(self):
        # If model is loaded from file-system
        if self.load_model:
            if self.path_to_model is None:
                logger.error(
                    "Path has not been set.",
                    next_step="Model will not be initialized.",
                    tip="Set a path_to_model in your config."
                )
                raise Exception

            if not os.path.exists(f"results/weights/{self.path_to_model}"):
                logger.error(
                    f"Can't find model at results/{self.path_to_model}.",
                    next_step="Model will not be initialized.",
                    tip=f"Check if the directory results/{self.path_to_model} exists."
                )
                raise Exception

            logger.info(f"Initializing model from {self.path_to_model}")
            return Db_vae.init(path_to_model=self.path_to_model, device=self.device, z_dim=self.z_dim).to(self.device)

        # Model is newly initialized
        logger.info(f"Creating new model with the following parameters:\n"
                    f"z_dim: {self.z_dim}\n"
                    f"hist_size: {self.hist_size}\n"
                    f"alpha: {self.alpha}\n"
                    f"num_bins: {self.num_bins}\n")

        return Db_vae(
            args=self.args,
            z_dim=self.z_dim,
            hist_size=self.hist_size,
            alpha=self.alpha,
            num_bins=self.num_bins,
            device=self.device
        ).to(device=self.device)

    def plot_loss(self, train_loss_values, val_loss_values):
        plt.plot(train_loss_values, label='train_loss')
        plt.plot(val_loss_values, label='val_loss')
        plt.xlabel("epoch")
        plt.legend(loc='upper left')
        plt.title('loss curve')
        # save image
        plt.savefig(f"results/logs/{self.args.test_no}/loss_curve.png")
        plt.clf()

    def plot_loss_recon(self, train_loss_recon_values):
        plt.plot(train_loss_recon_values, label='train_loss_recon')
        plt.xlabel("epoch")
        plt.legend(loc='upper left')
        plt.title('loss curve')
        # save image
        plt.savefig(f"results/logs/{self.args.test_no}/loss_recon_curve.png")
        plt.clf()

    def train(self, epochs: Optional[int] = None):
        # Optionally use passed epochs
        epochs = self.epochs if epochs is None else epochs

        train_loss_values = []
        val_loss_values = []
        train_loss_recon_values = []

        # Start training and validation cycle
        for epoch in range(epochs):
            epoch_start_t = datetime.datetime.now()
            logger.info(f"Starting epoch: {epoch+1}/{epochs}")

            self._update_sampling_histogram(epoch)

            # Training
            train_loss, train_acc, train_loss_recon = self._train_epoch(epoch)
            train_loss_values.append(train_loss)
            train_loss_recon_values.append(train_loss_recon)
            epoch_train_t = datetime.datetime.now() - epoch_start_t
            logger.info(f"epoch {epoch+1}/{epochs}::Training done")
            logger.info(f"epoch {epoch+1}/{epochs} => train_loss={train_loss:.2f}, train_acc={train_acc:.2f}")

            # Validation
            logger.info("Starting validation")
            val_loss, val_acc = self._eval_epoch(epoch)
            val_loss_values.append(val_loss)
            epoch_val_t = datetime.datetime.now() - epoch_start_t
            logger.info(f"epoch {epoch+1}/{epochs}::Validation done")
            logger.info(f"epoch {epoch+1}/{epochs} => val_loss={val_loss:.2f}, val_acc={val_acc:.2f}")

            # Print reconstruction
            self.print_reconstruction(self.model, self.train_loader.dataset, epoch, self.device)

            # Save model and scores
            # if epoch+1 == self.args.epochs:
            self._save_epoch(epoch, train_loss, val_loss, train_acc, val_acc, train_loss_recon)

        self.plot_loss(train_loss_values, val_loss_values)
        self.plot_loss_recon(train_loss_recon_values)

        logger.success(f"Finished training on {epochs} epochs.")

    # Outputting reconstructed input
    def print_reconstruction(self, model, data, epoch, device, n_rows=4, save=True):
        # TODO: Add annotation
        model.eval()
        n_samples = n_rows**2
        images = sample_dataset(data, n_samples).to(device)
        recon_images = model.recon_images(images)
        fig = plt.figure(figsize=(16, 8))

        fig.add_subplot(1, 2, 1)
        grid = make_grid(images.reshape(n_samples, 3, self.args.image_size, self.args.image_size), n_rows)
        plt.imshow(grid.permute(1, 2, 0).cpu())

        utils.remove_frame(plt)

        fig.add_subplot(1, 2, 2)
        grid = make_grid(recon_images.reshape(n_samples, 3, self.args.image_size, self.args.image_size), n_rows)
        plt.imshow(grid.permute(1, 2, 0).cpu())

        utils.remove_frame(plt)

        if save:
            fig.savefig(f'results/plots/{self.args.test_no}/reconstructions/epoch_{epoch+1}.png',
                        bbox_inches='tight', dpi=128)

            plt.close()
        else:
            return fig

    def plot_loop_perturb(self, var):
        self.model.eval()
        steps = 8
        data = self.valid_loader.dataset
        images = torch.stack([data[self.args.interp1][0], data[self.args.interp2][0]])
        recon_images = self.model.perturb_var(images, steps, var)
        fig = plt.figure(figsize=(16, 4))

        fig.add_subplot(1, steps + 2, 1)
        grid = make_grid(images[0].reshape(1, 3, self.args.image_size, self.args.image_size), 1)
        plt.gca().set_title('source\nimage', fontsize=16)
        plt.imshow(grid.permute(1, 2, 0).cpu())
        utils.remove_frame(plt)

        for i, im in enumerate(recon_images):
            fig.add_subplot(1, steps + 2, i + 2)
            grid = make_grid(recon_images[i].reshape(1, 3, self.args.image_size, self.args.image_size), 1)
            plt.imshow(grid.permute(1, 2, 0).cpu())
            utils.remove_frame(plt)

        fig.add_subplot(1, steps + 2, steps + 2)
        grid = make_grid(images[1].reshape(1, 3, self.args.image_size, self.args.image_size), 1)
        plt.gca().set_title('target\nimage', fontsize=16)
        plt.imshow(grid.permute(1, 2, 0).cpu())
        utils.remove_frame(plt)

        fig.savefig(f'results/plots/{self.args.test_no}/reconstructions/perturbations/perturb_{self.args.interp1}-{self.args.interp2}_var{var}'
                    f'.png', bbox_inches='tight', dpi=300)
        plt.close()

    def perturb(self):
        # printing out variable indexes to allow selection
        # with open(f'results/logs/{self.args.test_no}/variable_idxs.pkl', 'rb') as f:
        #     # variable_idxs = pickle.load(f)
        txt = Path(f'results/logs/{self.args.test_no}/variable_idxs.txt').read_text()
        txt = txt.replace('\n', ',')
        variable_idxs = txt.split(',')
        variable_idxs = [int(i) for i in variable_idxs[:-1]]

        for i, idx in enumerate(variable_idxs):
            print(i, idx)

        if args.perturb_single:
            var = variable_idxs[self.args.var_to_perturb]
            self.plot_loop_perturb(var)

        else:
            for var in tqdm(variable_idxs[:self.args.var_to_perturb]):
                self.plot_loop_perturb(var)

    def interpolate(self):
        self.model.eval()
        data = self.valid_loader.dataset
        steps = 8
        images = torch.stack([data[self.args.interp1][0], data[self.args.interp2][0]])
        recon_images = self.model.interpolate(images, steps)
        fig = plt.figure(figsize=(16, steps))
        for i, im in enumerate(recon_images):
            fig.add_subplot(1, steps + 1, i + 1)
            grid = make_grid(recon_images[i].reshape(1, 3, self.args.image_size, self.args.image_size), 1)
            plt.imshow(grid.permute(1, 2, 0).cpu())
            utils.remove_frame(plt)
        fig.savefig(f'results/plots/{self.args.test_no}/reconstructions/interpolate'
                    f'.png', bbox_inches='tight', dpi=128)
        plt.close()

    def _save_epoch(self, epoch: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float, train_loss_recon: float):
        """Writes training and validation scores to a csv, and stores a model to disk."""
        if not self.args.test_no:
            logger.warning(f"`--run_folder` could not be found.",
                           f"The program will continue, but won't save anything",
                           f"Double-check if --run_folder is configured.")

            return

        # Write epoch metrics
        path_to_results = f"results/logs/{self.args.test_no}/training_results.csv"
        with open(path_to_results, "a") as wf:
            wf.write(f"{epoch+1}, {train_loss}, {val_loss}, {train_acc}, {val_acc}, {train_loss_recon}\n")

        # Write model to disk
        path_to_model = f"results/weights/{self.args.test_no}/model.pt"
        torch.save(self.model.state_dict(), path_to_model)

        logger.save(f"Stored model and results at results/{self.args.test_no}")

    def visualize_bias(self, probs, data_loader, all_labels, all_index, epoch, n_rows=3):
        # TODO: Add annotation
        n_samples = n_rows ** 2
        # getting images with highest and lowest probability of being sampled
        highest_probs_idx = probs.argsort(descending=True)[:n_samples]
        lowest_probs_idx = probs.argsort()[:n_samples]
        # Getting mean sampling probability
        mean_probs = [np.mean(probs[highest_probs_idx].detach().cpu().numpy()),
                      np.mean(probs[lowest_probs_idx].detach().cpu().numpy())]

        highest_imgs = sample_idxs_from_loader(all_index[highest_probs_idx], data_loader, 1)
        lowest_imgs = sample_idxs_from_loader(all_index[lowest_probs_idx], data_loader, 1)

        img_list = (highest_imgs, lowest_imgs)
        titles = ("Lowest Representation", "Highest Representation")
        fig = plt.figure(figsize=(16, 16))

        for i in range(2):
            ax = fig.add_subplot(1, 2, i+1)
            grid = make_grid(img_list[i].reshape(n_samples, 3, self.args.image_size, self.args.image_size), n_rows)
            plt.imshow(grid.permute(1, 2, 0).cpu())
            # ax.set_title(f'{titles[i]}\nMean Sample Prob: {Decimal(str(mean_probs[i])):.3e}', fontdict={"fontsize":18}, pad=20)
            ax.set_title(f'{titles[i]}', fontdict={"fontsize": 24}, pad=20)

            utils.remove_frame(plt)

        path_to_results = f"results/plots/{self.args.test_no}/bias_probs/epoch_{epoch+1}.png"
        logger.save(f"Saving a bias probability figure in {path_to_results}")

        fig.savefig(path_to_results, bbox_inches='tight', dpi=128)
        plt.close()

    def _eval_epoch(self, epoch):
        """Calculates the validation error of the model."""

        self.model.eval()
        avg_loss = 0
        avg_acc = 0

        all_labels = torch.tensor([], dtype=torch.long).to(self.device)
        all_preds = torch.tensor([]).to(self.device)
        all_idxs = torch.tensor([], dtype=torch.long).to(self.device)

        count = 0

        with torch.no_grad():
            for i, (images, labels, idxs) in enumerate(self.valid_loader):

                images = images.to(self.device)
                labels = labels.to(self.device)
                idxs = idxs.to(self.device)
                pred, loss, sigout, _ = self.model.forward(images, labels)

                loss = loss.mean()
                acc = utils.calculate_accuracy(labels, pred)

                avg_loss += loss.item()
                avg_acc += acc

                all_labels = torch.cat((all_labels, labels))
                all_preds = torch.cat((all_preds, pred))
                all_idxs = torch.cat((all_idxs, idxs))

                count = i

        best_mel, worst_mel, best_ben, worst_ben = utils.get_best_and_worst_predictions(all_labels, all_preds, self.device)
        self.visualize_best_and_worst(self.valid_loader, all_labels, all_idxs, epoch, best_mel, worst_mel, best_ben, worst_ben)

        return avg_loss/(count+1), avg_acc/(count+1)

    def _train_epoch(self, epoch):
        """Trains the model for one epoch."""
        self.model.train()

        avg_loss: float = 0
        avg_loss_recon: float = 0
        avg_acc: float = 0
        count: int = 0

        # # KL-annealing
        # if epoch <5:
        #     c3 = np.linspace(0.01, 0.1, num=5)[epoch]
        # else:
        #     c3 = 0.1
        #     c3 = 0.1
        c3 = 0.1

        train_loss = []

        bar = tqdm(self.train_loader)
        for i, (images, labels, _) in enumerate(bar):
            # Forward pass
            images, labels = images.to(self.device), labels.to(self.device)  # sending data to GPU
            pred, loss, sigout, loss_recon = self.model.forward(images, labels, c3)

            # Calculate the gradient, and clip at 5
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss_recon = loss_recon.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
            self.optimizer.step()

            # Calculate metrics
            acc = utils.calculate_accuracy(labels, pred)
            avg_loss += loss.item()
            avg_loss_recon += loss_recon.item()
            avg_acc += acc

            if i == self.train_len/self.args.batch_size:
                logger.info(f"Training: batch:{i} accuracy:{acc}")

            loss_np = loss.detach().cpu().numpy()  # sending loss to cpu
            train_loss.append(loss_np)  # appending loss to loss list
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)  # calculating smooth loss

            count = i
            bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))  # metrics for loading bar

        return avg_loss/(count+1), avg_acc/(count+1), avg_loss_recon/(count+1)

    def _update_sampling_histogram(self, epoch: int):
        """Updates the data loader for faces to be proportional to how challenge each image is, in case
        debias_type not none is.
        """
        dataset_train = GenericImageDataset(csv=self.df_train)
        hist_loader = make_hist_loader(dataset_train, self.batch_size)

        if self.debias_type != 'none':
            hist = self._update_histogram(hist_loader, epoch)
            self.train_loader.sampler.weights = hist
        else:
            self.train_loader.sampler.weights = torch.ones(self.hist_size)

    def _update_histogram(self, data_loader, epoch):
        """Updates the histogram of `self.model`."""
        logger.info(f"Updating weight histogram using method: {self.debias_type}")

        self.model.means = torch.Tensor().to(self.device)
        self.model.std = torch.Tensor().to(self.device)

        all_labels = torch.tensor([], dtype=torch.long).to(self.device)
        all_index = torch.tensor([], dtype=torch.long).to(self.device)

        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                images, labels, index = batch
                images, labels, index = images.to(self.device), labels.to(self.device), index.to(self.device)

                all_labels = torch.cat((all_labels, labels))
                all_index = torch.cat((all_index, index))

                if self.debias_type == "max" or self.debias_type == "max5" or self.debias_type == "max50":
                    self.model.build_means(images)

                elif self.debias_type == "gaussian":
                    self.model.build_histo(images)

            if self.debias_type == "max":
                probs = self.model.get_histo_max()
            elif self.debias_type == "max5":
                probs = self.model.get_histo_max5()
            elif self.debias_type == "max50":
                probs = self.model.get_histo_max50()
            elif self.debias_type == "gaussian":
                probs = self.model.get_histo_gaussian()
            else:
                logger.error("No correct debias method given!",
                             next_step="The program will now close",
                             tip="Set --debias_method to 'max', 'max5' or 'gaussian'.")
                raise Exception

        self.visualize_bias(probs, data_loader, all_labels, all_index, epoch)

        return probs

    def sample(self, n_rows=4):
        n_samples = n_rows**2
        sample_images = self.model.sample(n_samples=n_samples)

        plt.figure(figsize=(n_rows*2, n_rows*2))
        grid = make_grid(sample_images.reshape(n_samples, 3, self.args.image_size, self.args.image_size), n_rows)
        plt.imshow(grid.permute(1, 2, 0).cpu())

        utils.remove_frame(plt)
        plt.show()

        return

    # def reconstruction_samples(self, n_rows=4):
    #     valid_data = concat_datasets(self.valid_loader.faces.dataset, self.valid_loader.nonfaces.dataset, proportion_a=0.5)
    #     fig = self.print_reconstruction(self.model, valid_data, 0, self.device, save=False)
    #
    #     fig.show()
    #
    #     return

    def visualize_best_and_worst(self, data_loader, all_labels, all_indices, epoch, best_mel, worst_mel, best_ben,
                                 worst_ben, n_rows=4, save=True):
        # TODO: Add annotation
        n_samples = n_rows**2

        fig = plt.figure(figsize=(16, 16))

        sub_titles = ["Best melanoma", "Worst melanoma", "Best benign", "Worst benign"]
        for i, indices in enumerate((best_mel, worst_mel, best_ben, worst_ben)):
            labels, indices = all_labels[indices], all_indices[indices]
            images = sample_idxs_from_loader(indices, data_loader, labels[0])

            ax = fig.add_subplot(2, 2, i+1)
            grid = make_grid(images.reshape(n_samples, 3, self.args.image_size, self.args.image_size), n_rows)
            plt.imshow(grid.permute(1, 2, 0).cpu())
            ax.set_title(sub_titles[i], fontdict={"fontsize": 30})

            utils.remove_frame(plt)

        if save:
            fig.savefig(f'results/plots/{self.args.test_no}/best_and_worst/epoch_{epoch+1}.png', bbox_inches='tight', dpi=128)

            plt.close()

        # else:
        #
        return fig

    def best_and_worst(self, n_rows=4):
        """Calculates the validation error of the model."""
        # face_loader, nonface_loader = self.valid_loader

        self.model.eval()
        avg_loss = 0
        avg_acc = 0

        all_labels = torch.tensor([], dtype=torch.long).to(self.device)
        all_preds = torch.tensor([]).to(self.device)
        all_idxs = torch.tensor([], dtype=torch.long).to(self.device)

        count = 0

        with torch.no_grad():
            for i, batch in enumerate(self.valid_loader):
                images, labels, idxs = batch

                images = images.to(self.device)
                labels = labels.to(self.device)
                idxs = idxs.to(self.device)
                pred, loss, _, _ = self.model.forward(images, labels)

                loss = loss.mean()
                acc = utils.calculate_accuracy(labels, pred)

                avg_loss += loss.item()
                avg_acc += acc

                all_labels = torch.cat((all_labels, labels))
                all_preds = torch.cat((all_preds, pred))
                all_idxs = torch.cat((all_idxs, idxs))

                count = i

        best_mal, worst_mal, best_ben, worst_ben = utils.get_best_and_worst_predictions(all_labels, all_preds, self.device)
        fig = self.visualize_best_and_worst(self.valid_loader, all_labels, all_idxs, 0, best_mal, worst_mal, best_ben, worst_ben, save=True)

        fig.show()

        return
