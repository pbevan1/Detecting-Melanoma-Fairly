import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil
from zipfile import ZipFile

# Detecting Gentian Violet Markers

# defining numpy arrays for HSV threshold values to look for in images
lower_violet = np.array([125, 100, 60], dtype=np.uint8)
upper_violet = np.array([145, 255, 255], dtype=np.uint8)

folder = '/Data/train'
# Looping through images to identify those with gentian violet pixels
for im in os.listdir(folder):
    src = f'/Data/train/{im}'
    img = cv2.imread(src)  # Reading image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Converting image to HSV for more effective trhesholding
    array = cv2.inRange(img, lower_violet, upper_violet)  # Creating array of pixels that fit within the threshold
    if 255 in array:  # Checking if the array contains any values of 255 (within the HSV values above)
        shutil.copy(src, '/Data/Train_Marked_OpenCV2')  # Copying to new directory for inspection

# Manually weed out anomolies by looking through 'marked' images

# Making list of the remaining images with gentian violet markers
marked_list = []
for i in os.listdir('/Data/Train_Marked_OpenCV2'):
    marked_list.append(str(i)[:12])
train = pd.read_csv(r'/Data/train.csv')  # Opening metadata/labels
train['marked'] = 0  # Creating 'marked' column and setting the default to 0 (False)
train.loc[train.image_name.isin(marked_list), 'marked'] = 1  # Setting images identified as marked to 1 (True)

# Manually labeled scale data

# Making list of the images with scales
scale_list = []
scale_images_path = '/content/drive/MyDrive/MSc Project/Data/train_512/train_scale'
for i in os.listdir(scale_images_path):
    scale_list.append(str(i)[:12])
train['scale'] = 0  # Creating 'scale' column and setting the default to 0 (False)
train.loc[train.image_name.isin(scale_list), 'scale'] = 1  # Setting images identified as having a scale to 1 (True)
train.to_csv('/content/drive/MyDrive/MSc Project/Data/train.csv', index=False)  # Saving the metadata/labels file with new columns
