import os
import pandas as pd
import cv2
from PIL import Image
import math
from skimage import io, color
import torch


# Get mean and standard deviation of training data for normalisation
def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    return mean.tolist(), std.tolist()


# Centre crops and resizes images and saves to new location
def crop_resize(rootdir, savedir, size):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            im = cv2.imread(os.path.join(subdir, file))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # if image wider than long centre crop by smallest side
            if im.shape[1] < im.shape[0]:
                dim1 = int(im.shape[1])
                dim0 = int(im.shape[0])
                chi = int((dim0 - dim1) / 2)
                im = im[chi:chi + dim1, 0:dim1]
            else:
                # if image longer than wide centre crop by smallest side
                dim1 = int(im.shape[0])
                dim0 = int(im.shape[1])
                chi = int((dim0 - dim1) / 2)
                im = im[0:dim1, chi:chi + dim1]

            im = Image.fromarray(im, 'RGB')
            # resizing
            im = im.resize((size, size), Image.ANTIALIAS)
            im.save(os.path.join(savedir, file))
    print("Done")


# Calculates image dimensions from raw images and saves to dataframe
def get_size_from_raw(df):
    def sizeify(filepath):
        image = Image.open(filepath)
        width, height = image.size
        return f'{width}x{height}'
    df['size'] = df['filepath'].apply(lambda x: sizeify(x))
    return df


# Use on ISIC dataframe to get dimensions as single variable
def get_size_ISIC(df):
    df['size'] = 0

    def sizeify(width, height):
        return f'{width}x{height}'

    df['size'] = df.apply(lambda x: sizeify(df.width, df.height), axis=1)
    return df


# Gets sizes for Fitzpatrick17k dataset and saves to csv
def get_fitzpatrick17_sizes():
    df_train = pd.read_csv(f'data/csv/fitzpatrick17k.csv')
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
    df_train['filepath'] = df_train['image_name'].apply(lambda x: f'data/raw_images/fitzpatrick17k/{x}')

    df_train = get_size_from_raw(df_train)
    df_train.to_csv('data/csv/fitzpatrick17k.csv')
    print('size column added to fitzpatrick17k.csv')


# Hair removal for ITA calculation
def hair_remove(image):
    # Convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Kernel for morphologyEx
    kernel = cv2.getStructuringElement(1, (17, 17))
    # Apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # Apply thresholding to blackhat
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Inpaint with original image and threshold image
    final_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
    return final_image


# Calculates Fitzpatrick skin type of an image using Kinyanjui et al.'s thresholds
def get_sample_ita_kin(path):
    ita_bnd_kin = -1
    try:
        rgb = io.imread(path)
        rgb = hair_remove(rgb)
        lab = color.rgb2lab(rgb)
        ita_lst = []
        ita_bnd_lst = []

        # Taking samples from different parts of the image
        L1 = lab[230:250, 115:135, 0].mean()
        b1 = lab[230:250, 115:135, 2].mean()

        L2 = lab[5:25, 115:135, 0].mean()
        b2 = lab[5:25, 115:135, 2].mean()

        L3 = lab[115:135, 5:25, 0].mean()
        b3 = lab[115:135, 5:25, 2].mean()

        L4 = lab[115:135, 230:250, 0].mean()
        b4 = lab[115:135, 230:250, 2].mean()

        L5 = lab[216:236, 216:236, 0].mean()
        b5 = lab[216:236, 216:236, 2].mean()

        L6 = lab[216:236, 20:40, 0].mean()
        b6 = lab[216:236, 20:40, 2].mean()

        L7 = lab[20:40, 20:40, 0].mean()
        b7 = lab[20:40, 20:40, 2].mean()

        L8 = lab[20:40, 216:236, 0].mean()
        b8 = lab[20:40, 216:236, 2].mean()

        L_lst = [L1, L2, L3, L4, L5, L6, L7, L8]
        b_lst = [b1, b2, b3, b4, b5, b6, b7, b8]

        # Calculating ITA values
        for L, b in zip(L_lst, b_lst):
            ita = math.atan((L - 50) / b) * (180 / math.pi)
            ita_lst.append(ita)

        # Using max ITA value (lightest)
        ita_max = max(ita_lst)

        # Getting skin shade band from ITA
        if ita_max > 55:
            ita_bnd_kin = 1
        if 41 < ita_max <= 55:
            ita_bnd_kin = 2
        if 28 < ita_max <= 41:
            ita_bnd_kin = 3
        if 19 < ita_max <= 28:
            ita_bnd_kin = 4
        if 10 < ita_max <= 19:
            ita_bnd_kin = 5
        if ita_max <= 10:
            ita_bnd_kin = 6
    except Exception:
        pass

    return ita_bnd_kin


# Getting skin types for ISIC data
def get_isic_skin_type():
    # Getting ITA for ISIC Training and saving to csv
    df_train = pd.read_csv('data/csv/isic_train_20-19-18-17.csv')
    df_train.loc[(df_train.year == 2020), 'filepath'] = df_train['image_name'].apply(
        lambda x: os.path.join(f'data/images/isic_20_train_256/{x}.jpg'))
    df_train.loc[(df_train.year != 2020), 'filepath'] = df_train['image_name'].apply(
        lambda x: os.path.join(f'data/images/isic_19_train_256', f'{x}.jpg'))

    df_train['fitzpatrick'] = df_train['filepath'].apply(lambda x: get_sample_ita_kin(x))
    df_train.to_csv('data/csv/isic_train_20-19-18-17.csv', index=False)
    print('Fitzpatrick skin type column added to isic_train_20-19-18-17.csv')


# Getting skin types for Fitzpatrick17k data (and compare to human labelled for accuracy)
def get_fitz17k_skin_type():
    # Getting ITA for Fitzpatrick17k and saving to csv
    df_fitz = pd.read_csv('data/csv/fitzpatrick17k.csv')
    df_fitz = df_fitz.loc[(df_fitz.qc != '3 Wrongly labelled'), :]
    df_fitz = df_fitz.loc[df_fitz['url'].str.contains('http', na=False), :]
    df_fitz = df_fitz.loc[df_fitz['fitzpatrick'] != -1, :]
    df_fitz = pd.get_dummies(df_fitz, columns=['three_partition_label'], drop_first=True)
    df_fitz.rename(columns={'three_partition_label_malignant': 'target'}, inplace=True)
    df_fitz['image_name'] = 0
    for i, url in enumerate(df_fitz.url):
        if 'atlasderm' in url:
            df_fitz.loc[df_fitz['url'] == url, 'image_name'] = f'atlas{i}.jpg'
        else:
            df_fitz.loc[df_fitz['url'] == url, 'image_name'] = url.split('/', -1)[-1]
    # Adding column with path to file
    df_fitz['filepath'] = df_fitz['image_name'].apply(lambda x: f'data/images/fitzpatrick17k_256/{x}')

    df_fitz['pred_kin'] = df_fitz['filepath'].apply(lambda x: get_sample_ita_kin(x))

    df_fitz.to_csv('data/csv/output_csv/fitzpatrick17k.csv', index=False)
    print('Fitzpatrick skin type column added to fitzpatrick17k.csv')

    df_fitz = pd.read_csv('data/csv/output_csv/fitzpatrick17k.csv')
    df_fitz = df_fitz.loc[df_fitz['fitzpatrick'] != -1, :]
    # Used to calculate if within + or - 1 as per Fitzpatrick17k paper
    df_fitz['pred_kin_plus1'] = df_fitz.pred_kin + 1
    df_fitz['pred_kin_minus1'] = df_fitz.pred_kin - 1

    kin_correct = sum(sum([df_fitz.pred_kin == df_fitz.fitzpatrick, df_fitz.pred_kin_plus1 == df_fitz.fitzpatrick,
                           df_fitz.pred_kin_minus1 == df_fitz.fitzpatrick]))

    print(f'Accuracy using Kinyanjui ITA thresholds: {round(kin_correct/len(df_fitz)*100,2)}%')


if __name__ == '__main__':
    # Getting skin type of ISIC data using labelling algorithm and saving to column 'fitzpatrick'
    get_isic_skin_type()

    # Getting skin type of Fitzpatrick17k data using labelling algorithm and saving to column 'fitzpatrick'.
    # Also prints accuracy scores vs human labels using Kinyanjui thresholds.
    get_fitz17k_skin_type()
