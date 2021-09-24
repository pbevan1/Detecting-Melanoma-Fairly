from typing import List
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import ConcatDataset, DataLoader
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import os
import uuid
from typing import Optional, NamedTuple
import gc
from collections import Counter
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix
from DB_VAE.logger import logger


# inverse transform to get normalize image back to original form for visualization
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)


# Default transform
default_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


class DatasetOutput(NamedTuple):
    image: torch.FloatTensor
    label: int
    idx: int


def visualize_tensor(img_tensor: torch.Tensor):
    pil_transformer = transforms.ToPILImage()
    pil_transformer(img_tensor).show()


def save_images(torch_tensors: torch.Tensor, path_to_folder: str):
    rand_filenames = str(uuid.uuid4())[:8]
    pil_transformer = transforms.ToPILImage()
    image_folder = f"results/{path_to_folder}/debug/images/{rand_filenames}/"
    os.makedirs(image_folder, exist_ok=True)

    for i, img in enumerate(torch_tensors):
        pil_img = pil_transformer(img)
        pil_img.save(f"{image_folder}/{rand_filenames}_{i}.jpg")

    return torch_tensors


def calculate_accuracy(labels, pred):
    """Calculates accuracy given labels and predictions."""
    return float(((pred > 0) == (labels > 0)).sum()) / labels.size()[0]


def calculate_AUC(labels, sigout):
    return roc_auc_score((labels == 1).astype(float), sigout[:, 1])


def calculate_sens_spec(labels, sigout):
    cm = confusion_matrix((labels == 1).astype(float), sigout[:, 1])  # Defining confusion matrix
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]  # getting confusion matrix values
    sensitivity = tp / (tp + fn)  # calculating sensitivity
    specificity = tn / (tn + fp)  # calculating specificity
    return sensitivity, specificity


def get_best_and_worst_predictions(labels, pred, device):
    """Returns indices of the best and worst predicted images."""
    n_rows = 4
    n_samples = n_rows**2

    logger.info(f"Melanoma percentage: {float(labels.sum().item())/len(labels)}")
    indices = torch.tensor([i for i in range(len(labels))]).long().to(device)

    melslice = labels == 1
    mel, ben = pred[melslice],    pred[~melslice]
    melanoma_index, benign_index = indices[melslice], indices[~melslice]

    worst_mal = melanoma_index[mel.argsort()[:n_samples]]
    best_mal = melanoma_index[mel.argsort(descending=True)[:n_samples]]

    worst_ben = benign_index[ben.argsort(descending=True)[:n_samples]]
    best_ben = benign_index[ben.argsort()[:n_samples]]

    return best_mal, worst_mal, best_ben, worst_ben


def calculate_places(name_list, setups, w, s):
    """Calculates the places in the final barplot."""
    x_axis = np.arange(len(setups))
    counter = len(name_list)-1

    if (len(name_list) % 2) == 0:
        places = []
        times = 0
        while counter > 0:
            places.append(x_axis-(s/2)-s*times)
            places.append(x_axis+(s/2)+s*times)

            times += 1
            counter -= 2

    else:
        places = [x_axis]
        times = 1
        while counter > 0:
            places.append(x_axis-s*times)
            places.append(x_axis+s*times)

            times += 1
            counter -= 2

    return x_axis, sorted(places, key=lambda sub: (sub[0], sub[0]))


def make_bar_plot(df, name_list, setups, colors=None, training_type=None, y_label="",
                  title="", y_lim=None, y_ticks=None):
    """Writes a bar plot for the final evaluation, based on the dataframe which stems from a results.csv."""
    if training_type is None:
        training_type = name_list
    if colors is None:
        colors = np.random.rand(len(name_list), 3)

    s = 0.8/len(name_list)
    w = s-0.02

    x_axis, places = calculate_places(name_list, setups, w, s)

    _ = plt.figure(figsize=(16, 6))
    ax = plt.subplot(111)
    for i in range(len(name_list)):
        ax.bar(places[i], df.loc[df["name"].str.contains(name_list[i]), setups].mean(), label=training_type[i],
               yerr=df.loc[df["name"].str.contains(name_list[i]), setups].std(), color=colors[i], width=w,
               edgecolor="black", linewidth=2, capsize=10)

    plt.ylabel(y_label, fontdict={"fontsize": 20})
    plt.xticks(x_axis, setups, fontsize=25)

    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])

    if y_ticks is not None:
        plt.yticks(y_ticks, fontsize=20)

    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=False, prop={'size': 19})
    plt.show()


def make_box_plot(df, name_list, training_type=None, colors=None, y_label="", title="", y_lim=None):
    """Writes a box plot for the final evaluation, based on the dataframe which stems from a results.csv."""
    if training_type is None:
        training_type = [""] + name_list

    fig = plt.figure(figsize=(16, 6))

    box_plot_data = [df.loc[df["name"].str.contains(name), :]['var'] for name in name_list]
    box = plt.boxplot(box_plot_data, patch_artist=True)

    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])

    plt.xticks(range(len(training_type)), training_type, fontsize=12)

    if colors is not None:
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

    plt.ylabel(y_label, fontsize=20)
    plt.show()


def remove_frame(plt):
    """Removes frames from a pyplot plot. """
    # TODO: Add annotation
    frame = plt.gca()
    for xlabel_i in frame.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    for xlabel_i in frame.axes.get_yticklabels():
        xlabel_i.set_fontsize(0.0)
        xlabel_i.set_visible(False)
    for tick in frame.axes.get_xticklines():
        tick.set_visible(False)
    for tick in frame.axes.get_yticklines():
        tick.set_visible(False)


def concat_batches(batch_a: DatasetOutput, batch_b: DatasetOutput):
    """Concatenates two batches of data of shape image x label x idx."""
    images: torch.Tensor = torch.cat((batch_a.image, batch_b.image), 0)
    labels: torch.Tensor = torch.cat((batch_a.label, batch_b.label), 0)
    idxs: torch.Tensor = torch.cat((batch_a.idx, batch_b.idx), 0)

    return images, labels, idxs


def read_image(path_to_image):
    """Reads an image into memory and transform to a tensor."""
    img: Image = Image.open(path_to_image)

    transforms = default_transforms()
    img_tensor: torch.Tensor = transforms(img)

    return img_tensor


def read_flags(path_to_model):
    """"""
    path_to_flags = f"results/{path_to_model}/flags.txt"

    with open(path_to_flags, 'r') as f:
        data = f.readlines()


def find_face_in_subimages(model, sub_images: torch.Tensor, device: str):
    """Finds a face in a tensor of subimages using a models' evaluation method."""
    model.eval()

    for images in sub_images:
        if len(images.shape) == 5:
            images = images.squeeze(dim=0)

        # If one image
        if len(images.shape) == 3:
            images = images.view(1, 3, 256, 256)
        images = images.to(device)
        pred = model.forward_eval(images)

        # If face
        if (pred > 0).any():
            return True

    return False


def default_transforms():
    """Transforms a transform object to a 256 by 256 tensor."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


def visualize_tensor(img_tensor: torch.Tensor):
    """Visualizes a image tensor."""
    pil_transformer = transforms.ToPILImage()
    pil_transformer(img_tensor).show()
