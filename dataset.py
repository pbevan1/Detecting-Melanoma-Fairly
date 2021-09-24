import os
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import albumentations as A
from sklearn.utils import shuffle
from globalbaz import args


# Dataset class, outputs data and labels as per arguments
class SIIMISICDataset(Dataset):
    def __init__(self, csv, split, mode, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.split = split
        self.mode = mode
        self.transform = transform
        self.args = args

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        if self.mode == 'test':
            return data, torch.tensor(self.csv.iloc[index].target).long()

        # Returning different data based on what test is being run
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long(), torch.tensor(
                    self.csv.iloc[index].fitzpatrick).long(),  torch.tensor(self.csv.iloc[index].fitzpatrick).long()


# Augmentations
def get_transforms():
    # Augmentations for all training data
    if args.arch == 'inception':  # Special augmentation for inception to provide 299x299 images
        transforms_train = A.Compose([
            A.Resize(299, 299),
            A.Normalize()
        ])
    else:
        transforms_train = A.Compose([
            A.Resize(args.image_size, args.image_size),
            A.Normalize()
        ])

    # Augmentations for validation data
    transforms_val = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize()
    ])
    return transforms_train, transforms_val


def get_df():
    # Loading test csvs
    df_test_atlasD = pd.read_csv(os.path.join(args.csv_dir, 'atlas_processed.csv'))
    df_test_atlasD['filepath'] = df_test_atlasD['derm'].apply(
        lambda x: f'{args.image_dir}/atlas_{args.image_size}/{x}')

    df_test_atlasC = pd.read_csv(os.path.join(args.csv_dir, 'atlas_processed.csv'))
    df_test_atlasC['filepath'] = df_test_atlasC['clinic'].apply(
        lambda x: f'{args.image_dir}/atlas_{args.image_size}/{x}')

    df_test_ASAN = pd.read_csv(os.path.join(args.csv_dir, 'asan.csv'))
    df_test_ASAN['filepath'] = df_test_ASAN['image_name'].apply(
        lambda x: f'{args.image_dir}/asan_{args.image_size}/{x}')

    df_test_MClassD = pd.read_csv(os.path.join(args.csv_dir, 'MClassD.csv'))
    df_test_MClassD['filepath'] = df_test_MClassD['image_name'].apply(
        lambda x: f'{args.image_dir}/MClassD_{args.image_size}/{x}')

    df_test_MClassC = pd.read_csv(os.path.join(args.csv_dir, 'MClassC.csv'))
    df_test_MClassC['filepath'] = df_test_MClassC['image_name'].apply(
        lambda x: f'{args.image_dir}/MClassC_{args.image_size}/{x}')

    # Placeholders for dataframes that are conditionally instantiated
    df_34 = []
    df_56 = []
    df_val = []

    if args.dataset == 'ISIC':
        # Loading train csv
        df_train = pd.read_csv(os.path.join(args.csv_dir, 'isic_train_20-19-18-17.csv'), low_memory=False)

        # Removing overlapping Mclass images from training data to prevent leakage
        df_train = df_train.loc[df_train.mclassd != 1, :]

        # Removing 2019 comp data from training data
        df_train = df_train.loc[df_train.year != 2019, :]
        df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)

        # Setting cv folds for 2017 data
        df_train.loc[(df_train.year != 2020) & (df_train.year != 2018), 'fold'] = df_train['tfrecord'] % 5
        tfrecord2fold = {
            2: 0, 4: 0, 5: 0,
            1: 1, 10: 1, 13: 1,
            0: 2, 9: 2, 12: 2,
            3: 3, 8: 3, 11: 3,
            6: 4, 7: 4, 14: 4,
        }
        # Setting cv folds for 2020 data
        df_train.loc[(df_train.year == 2020), 'fold'] = df_train['tfrecord'].map(tfrecord2fold)
        # Putting image filepath into column
        df_train.loc[(df_train.year == 2020), 'filepath'] = df_train['image_name'].apply(
            lambda x: os.path.join(f'{args.image_dir}/isic_20_train_{args.image_size}/{x}.jpg'))
        df_train.loc[(df_train.year != 2020), 'filepath'] = df_train['image_name'].apply(
            lambda x: os.path.join(f'{args.image_dir}/isic_19_train_{args.image_size}', f'{x}.jpg'))

        # Mapping fitzpatrick types to python range (from 0)
        fp2idx = {d: idx for idx, d in enumerate(sorted(df_train['fitzpatrick'].unique()))}
        df_train['fitzpatrick'] = df_train['fitzpatrick'].map(fp2idx)
        # Get validation set for hyperparameter tuning
        df_val = df_train.loc[df_train.year == 2018, :].reset_index()
        df_val['instrument'] = 0  # Adding instrument placeholder to prevent error
        _, df_val = train_test_split(df_val, test_size=0.33, random_state=args.seed, shuffle=True)
        # Removing val data from training set
        df_train = df_train.loc[df_train.year != 2018, :]

        if args.split_skin_types:
            # Splitting to test and train based on skin type groups
            df_34 = df_train.loc[(df_train['fitzpatrick'] == 2) | (df_train['fitzpatrick'] == 3), :]
            df_56 = df_train.loc[(df_train['fitzpatrick'] == 4) | (df_train['fitzpatrick'] == 5), :]
            df_train = df_train.loc[(df_train['fitzpatrick'] == 0) | (df_train['fitzpatrick'] == 1), :]
            df_train = df_train.sample(frac=1).reset_index(drop=True)
            df_train = shuffle(df_train)
            df_train['fold'] = 0
            # Setting up folds for cross validation
            len_df = len(df_train)
            df_train.iloc[int(len_df / 5 * 1):int(len_df / 5 * 2), df_train.columns.get_loc('fold')] = 1
            df_train.iloc[int(len_df / 5 * 2):int(len_df / 5 * 3), df_train.columns.get_loc('fold')] = 2
            df_train.iloc[int(len_df / 5 * 3):int(len_df / 5 * 4), df_train.columns.get_loc('fold')] = 3
            df_train.iloc[int(len_df / 5 * 4):, df_train.columns.get_loc('fold')] = 4

        if args.instrument:
            # Keeping only most populated groups of image sizes to use as proxy for instruments
            keep = ['6000x6000', '1872x1872', '640x640', '5184x5184', '1024x1024',
                    '3264x3264', '4288x4288', '2592x2592']
            df_train = df_train.loc[df_train['size'].isin(keep), :]
            # mapping image size to index as proxy for instrument
            size2idx = {d: idx for idx, d in enumerate(sorted(df_train['size'].unique()))}
            df_train['instrument'] = df_train['size'].map(size2idx)

    elif args.dataset == 'Fitzpatrick17k':
        # Loading fitzpatrick17k as train csv
        df_train = pd.read_csv(f'{args.csv_dir}/fitzpatrick17k.csv')
        # df_train['fitzpatrick'] = df_train['fitzpatrick'].astype(np.float32)
        # Discarding non-neoplastic and wrongly labelled data
        df_train = df_train.loc[
                  (df_train.three_partition_label != 'non-neoplastic') & (df_train.qc != '3 Wrongly labelled'), :]
        # Getting only downloadable data
        df_train = df_train.loc[df_train['url'].str.contains('http', na=False), :]
        df_train = df_train.loc[df_train['fitzpatrick'] != -1, :]
        # Creating benign/malignant labels
        df_train = pd.get_dummies(df_train, columns=['three_partition_label'], drop_first=True)
        df_train.rename(columns={'three_partition_label_malignant': 'target'}, inplace=True)
        df_train['image_name'] = 0
        for i, url in enumerate(df_train.url):
            if 'atlasderm' in url:
                df_train.loc[df_train['url'] == url, 'image_name'] = f'atlas{i}.jpg'
            else:
                df_train.loc[df_train['url'] == url, 'image_name'] = url.split('/', -1)[-1]
        # Adding column with path to file
        df_train['filepath'] = df_train['image_name'].apply(lambda x: f'{args.image_dir}/fitzpatrick17k_{args.image_size}/{x}')
        # Mapping fitzpatrick image to class index
        fp2idx = {d: idx for idx, d in enumerate(sorted(df_train['fitzpatrick'].unique()))}
        df_train['fitzpatrick'] = df_train['fitzpatrick'].map(fp2idx)
        if args.split_skin_types:
            # Splitting to test and train
            df_34 = df_train.loc[(df_train['fitzpatrick'] == 2) | (df_train['fitzpatrick'] == 3), :]
            df_56 = df_train.loc[(df_train['fitzpatrick'] == 4) | (df_train['fitzpatrick'] == 5), :]
            df_train = df_train.loc[(df_train['fitzpatrick'] == 0) | (df_train['fitzpatrick'] == 1), :]
            df_train = df_train.sample(frac=1).reset_index(drop=True)
            df_train = shuffle(df_train)
            df_train['fold'] = 0
            # Setting up folds for cross validation
            len_df = len(df_train)
            df_train.iloc[int(len_df / 5 * 1):int(len_df / 5 * 2), df_train.columns.get_loc('fold')] = 1
            df_train.iloc[int(len_df / 5 * 2):int(len_df / 5 * 3), df_train.columns.get_loc('fold')] = 2
            df_train.iloc[int(len_df / 5 * 3):int(len_df / 5 * 4), df_train.columns.get_loc('fold')] = 3
            df_train.iloc[int(len_df / 5 * 4):, df_train.columns.get_loc('fold')] = 4

    mel_idx = 1  # Setting index for positive class

    # Returning training, validation and test datasets
    return df_train, df_val, df_test_atlasD, df_test_atlasC,\
        df_test_ASAN, df_test_MClassD, df_test_MClassC, df_34, df_56, mel_idx
