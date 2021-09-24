from torchvision.transforms import ToPILImage
from typing import Callable
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import DB_VAE.utils as utils
from DB_VAE.setup import *


class GenericImageDataset(Dataset):
    """Generic dataset which defines all basic operations for the images."""
    def __init__(
        self,
        csv,
        filter_skin_color=None,
        path_to_images: str = args.image_dir,
        transform: Callable = utils.default_transform,
        **kwargs
    ):
        self.filter_skin_color = filter_skin_color
        self.csv = csv.reset_index(drop=True)
        if filter_skin_color is not None:
            self.csv = self.csv.loc[self.csv.fitzpatrick == self.filter_skin_color, :]

        # self.store: pd.DataFrame = self.init_store(self.csv.filepath)
        self.path_to_images = path_to_images
        self.transform = transform

        self.pil_transformer = ToPILImage()

        # Create store for data
        self.store = None

    def __getitem__(self, idx: int):
        row = self.csv.iloc[idx]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            res = self.transform(image)
            image = res.numpy().astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).permute(1, 2, 0).float()

        return utils.DatasetOutput(
            image=data,
            label=torch.tensor(self.csv.iloc[idx].target).long(),
            idx=torch.tensor(idx).long(),
        )

    def read_image(self, idx: int):
        """Interface, returns an PIL Image using the index."""
        pass

    def _apply_filters_to_metadata(self):
        """Allows filters to filter out countries, skin-colors and genders."""
        result = self.csv
        result = result.loc[result.fitzpatrick != self.filter_skin_color, :]
        return result

    def __len__(self):
        return self.csv.shape[0]
