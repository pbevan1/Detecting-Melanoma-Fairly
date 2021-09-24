from enum import Enum
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, SequentialSampler
from torch.utils.data.dataset import Subset
import pandas as pd
from sklearn.model_selection import train_test_split
from DB_VAE.generic import *


def get_df():
    # Loading test csv
    df_test_atlasD = pd.read_csv(os.path.join(args.csv_dir, 'atlas_processed.csv'))
    # adding column with path to file
    df_test_atlasD['filepath'] = df_test_atlasD['derm'].apply(
        lambda x: f'{args.image_dir}/atlas_{args.image_size}/{x}')

    df_test_atlasC = pd.read_csv(os.path.join(args.csv_dir, 'atlas_processed.csv'))
    # adding column with path to file
    df_test_atlasC['filepath'] = df_test_atlasC['clinic'].apply(
        lambda x: f'{args.image_dir}/atlas_{args.image_size}/{x}')

    df_test_ASAN = pd.read_csv(os.path.join(args.csv_dir, 'asan.csv'))
    # adding column with path to file
    df_test_ASAN['filepath'] = df_test_ASAN['image_name'].apply(
        lambda x: f'{args.image_dir}/asan_{args.image_size}/{x}')

    df_test_MClassD = pd.read_csv(os.path.join(args.csv_dir, 'MClassD.csv'))
    # adding column with path to file
    df_test_MClassD['filepath'] = df_test_MClassD['image_name'].apply(
        lambda x: f'{args.image_dir}/MClassD_{args.image_size}/{x}')

    df_test_MClassC = pd.read_csv(os.path.join(args.csv_dir, 'MClassC.csv'))
    # adding column with path to file
    df_test_MClassC['filepath'] = df_test_MClassC['image_name'].apply(
        lambda x: f'{args.image_dir}/MClassC_{args.image_size}/{x}')

    # Placeholders for dataframes that are conditionally instantiated
    df_34 = []
    df_56 = []
    df_val = []
    # fitz_test = []

    if args.dataset == 'ISIC':
        # loading train csv
        df_train = pd.read_csv(os.path.join(args.csv_dir, 'isic_train_20-19-18-17.csv'), low_memory=False)

        # Removing Mclass images from training data to prevent leakage
        df_train = df_train.loc[df_train.mclassd != 1, :]

        # removing 2018 comp data from training data
        df_train = df_train.loc[df_train.year != 2019, :]
        df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)

        # setting cv folds for 2017 data
        df_train.loc[(df_train.year != 2020), 'fold'] = df_train['tfrecord'] % 5
        tfrecord2fold = {
            2: 0, 4: 0, 5: 0,
            1: 1, 10: 1, 13: 1,
            0: 2, 9: 2, 12: 2,
            3: 3, 8: 3, 11: 3,
            6: 4, 7: 4, 14: 4,
        }
        # setting cv folds for 2020 data
        df_train.loc[(df_train.year == 2020), 'fold'] = df_train['tfrecord'].map(tfrecord2fold)
        # Putting image filepath into column
        df_train.loc[(df_train.year == 2020), 'filepath'] = df_train['image_name'].apply(
            lambda x: os.path.join(f'{args.image_dir}/isic_20_train_{args.image_size}/{x}.jpg'))
        df_train.loc[(df_train.year != 2020), 'filepath'] = df_train['image_name'].apply(
            lambda x: os.path.join(f'{args.image_dir}/isic_19_train_{args.image_size}', f'{x}.jpg'))

        # Mapping fitzpatrick types to pythonic range
        fp2idx = {d: idx for idx, d in enumerate(sorted(df_train['fitzpatrick'].unique()))}
        df_train['fitzpatrick'] = df_train['fitzpatrick'].map(fp2idx)
        # Get validation set for hyperparameter tuning
        df_val = df_train.loc[df_train.year == 2018, :].reset_index()
        df_val['instrument'] = 0  # Adding instrument placeholder to prevent error
        _, df_val = train_test_split(df_val, test_size=0.33, random_state=42, shuffle=True)
        # Removing val data from training set
        df_train = df_train.loc[df_train.year != 2018, :]

        if args.split_skin_types:
            # Splitting to test and train
            df_34 = df_train.loc[(df_train['fitzpatrick'] == 2) | (df_train['fitzpatrick'] == 3), :]
            df_56 = df_train.loc[(df_train['fitzpatrick'] == 4) | (df_train['fitzpatrick'] == 5), :]
            df_train = df_train.loc[(df_train['fitzpatrick'] == 0) | (df_train['fitzpatrick'] == 1), :]
            df_train = df_train.sample(frac=1).reset_index(drop=True)
            df_train = shuffle(df_train)
            df_train['fold'] = 0
            len_df = len(df_train)
            df_train.iloc[int(len_df / 5 * 1):int(len_df / 5 * 2), df_train.columns.get_loc('fold')] = 1
            df_train.iloc[int(len_df / 5 * 2):int(len_df / 5 * 3), df_train.columns.get_loc('fold')] = 2
            df_train.iloc[int(len_df / 5 * 3):int(len_df / 5 * 4), df_train.columns.get_loc('fold')] = 3
            df_train.iloc[int(len_df / 5 * 4):, df_train.columns.get_loc('fold')] = 4

    # Loading fitzpatrick17k as train csv
    elif args.dataset == 'Fitzpatrick17k':
        df_train = pd.read_csv(f'{args.csv_dir}/fitzpatrick17k.csv')
        # df_train['fitzpatrick'] = df_train['fitzpatrick'].astype(np.float32)
        df_train = df_train.loc[
                  (df_train.three_partition_label != 'non-neoplastic') & (df_train.qc != '3 Wrongly labelled'), :]
        df_train = df_train.loc[df_train['url'].str.contains('http', na=False), :]
        df_train = df_train.loc[df_train['fitzpatrick'] != -1, :]
        df_train = pd.get_dummies(df_train, columns=['three_partition_label'], drop_first=True)
        df_train.rename(columns={'three_partition_label_malignant': 'target'}, inplace=True)
        df_train['image_name'] = 0
        for i, url in enumerate(df_train.url):
            if 'atlasderm' in url:
                df_train.loc[df_train['url'] == url, 'image_name'] = f'atlas{i}.jpg'
            else:
                df_train.loc[df_train['url'] == url, 'image_name'] = url.split('/', -1)[-1]
        # adding column with path to file
        df_train['filepath'] = df_train['image_name'].apply(lambda x: f'{args.image_dir}/fitzpatrick17k_{args.image_size}/{x}')
        # mapping fitzpatrick image to class index
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
            len_df = len(df_train)
            df_train.iloc[int(len_df / 5 * 1):int(len_df / 5 * 2), df_train.columns.get_loc('fold')] = 1
            df_train.iloc[int(len_df / 5 * 2):int(len_df / 5 * 3), df_train.columns.get_loc('fold')] = 2
            df_train.iloc[int(len_df / 5 * 3):int(len_df / 5 * 4), df_train.columns.get_loc('fold')] = 3
            df_train.iloc[int(len_df / 5 * 4):, df_train.columns.get_loc('fold')] = 4

    mel_idx = 1

    return df_train, df_val, df_test_atlasD, df_test_atlasC, df_test_ASAN, \
        df_test_MClassD, df_test_MClassC, df_34, df_56, mel_idx


class EvalDatasetType(Enum):
    """Defines a enumerator the makes it possible to double check dataset types."""
    PBB_ONLY = 'ppb'
    IMAGENET_ONLY = 'imagenet'
    H5_IMAGENET_ONLY = 'h5_imagenet'


def make_eval_loader(
    num_workers: int,
    csv,
    filter_skin_color=5,
    **kwargs
):
    """Creates an evaluaion data loader."""
    dataset = GenericImageDataset(csv=csv, filter_skin_color=filter_skin_color)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    return data_loader


def subsample_dataset(dataset: Dataset, nr_subsamples: int, random=False):
    """Create a specified number of subsamples from a dataset."""
    idxs = np.arange(nr_subsamples)

    if random:
        idxs = np.random.choice(np.arange(len(dataset)), nr_subsamples)

    return Subset(dataset, idxs)


def sample_dataset(dataset: Dataset, nr_samples: int):
    """Create a tensor stack of a specified number from a given dataset."""
    max_nr_items: int = min(nr_samples, len(dataset))
    idxs = np.random.permutation(np.arange(len(dataset)))[:max_nr_items]

    return torch.stack([dataset[idx][0] for idx in idxs])


def sample_idxs_from_loader(idxs, data_loader, label):
    """Returns data id's from a dataloader."""
    if label == 1:
        dataset = data_loader.dataset
    else:
        dataset = data_loader.dataset

    return torch.stack([dataset[idx.item()][0] for idx in idxs])


def make_hist_loader(dataset, batch_size):
    """Retrun a data loader that return histograms from the data."""
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

    return DataLoader(dataset, batch_sampler=batch_sampler)
