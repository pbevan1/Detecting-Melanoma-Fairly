import requests
import zipfile
import os
import pandas as pd
import shutil
import urllib
import gdown
from preprocessing import crop_resize

'''
Downloads, crops and resizes to 256x256 images from the following datasets:
>MClass Dermoscopic
>MClass Clinical
>ASAN
>Fitzpatrick17k
Raw images are downloaded temporarily and deleted once resizing has taken place
ISIC and Atlas data should be downloaded separately.
'''

shutil.rmtree('data/images')  # clearing images folder to make sure script runs if executed second time

# Creating directories for images to be downloaded
os.makedirs('data/images/', exist_ok=True)  # remaking images folder
os.makedirs('data/images/atlas_256/', exist_ok=True)
os.makedirs('data/images/MClassD_256/', exist_ok=True)
os.makedirs('data/images/MClassC_256/', exist_ok=True)
os.makedirs('data/images/asan_256/', exist_ok=True)
os.makedirs('data/raw_images/fitzpatrick17k/', exist_ok=True)
os.makedirs('data/images/fitzpatrick17k_256/', exist_ok=True)
print('Directories created')

# Downloading ISIC data
url20 = 'https://drive.google.com/uc?id=1BUoRe_0AABhlTOZdu_JONSuCzTgXqem6'
output20 = 'data/raw_images/isic_20_train_256.zip'
url19 = 'https://drive.google.com/uc?id=1fUqSJs_IO1MlmkS6MAp80IGQBZf7gggP'
output19 = 'data/raw_images/isic_19_train_256.zip'
gdown.download(url20, output20, quiet=False)
gdown.download(url19, output19, quiet=False)
print('ISIC images downloaded')

# Downloading zip file of raw MClass dermoscopic images
mclassd = requests.get("https://skinclass.de/MClass/MClass-D.zip")
path_to_mclassd = "./data/raw_images/MClassD_raw.zip"
file = open(path_to_mclassd, "wb")
file.write(mclassd.content)
file.close()
print('Raw MClass Dermoscopic images downloaded')

# Downloading zip file of raw MClass clinical images
mclassc = requests.get("https://skinclass.de/MClass/MClass-ND")
path_to_mclassc = "./data/raw_images/MClassC_raw.zip"
file = open(path_to_mclassc, "wb")
file.write(mclassc.content)
file.close()
print('Raw MClass Clinical images downloaded')

# Downloading zip file of raw ASAN images
ASAN = requests.get("https://ndownloader.figshare.com/files/9328573")
path_to_asan = "./data/raw_images/ASAN_raw.zip"
file = open(path_to_asan, "wb")
file.write(ASAN.content)
file.close()
print('Raw ASAN images downloaded')

# Unzipping Atlas, MClass and ASAN files
with zipfile.ZipFile('data/raw_images/isic_20_train_256.zip', 'r') as zip_ref:
    zip_ref.extractall('data/raw_images/isic_20_train_256')
with zipfile.ZipFile('data/raw_images/isic_19_train_256.zip', 'r') as zip_ref:
    zip_ref.extractall('data/raw_images/isic_19_train_256')
with zipfile.ZipFile('data/raw_images/release_v0.zip', 'r') as zip_ref:
    zip_ref.extractall('data/raw_images/Atlas')
with zipfile.ZipFile(path_to_mclassd, 'r') as zip_ref:
    zip_ref.extractall('data/raw_images/MClassD')
with zipfile.ZipFile(path_to_mclassc, 'r') as zip_ref:
    zip_ref.extractall('data/raw_images/MClassC')
with zipfile.ZipFile(path_to_asan, 'r') as zip_ref:
    zip_ref.extractall('data/raw_images/ASAN')

shutil.move('data/raw_images/isic_20_train_256/isic_20_train_256', 'data/images/')
shutil.move('data/raw_images/isic_19_train_256/isic_19_train_256', 'data/images/')

# Centre cropping and resizing to 256x256
crop_resize('data/raw_images/Atlas/release_v0/images', 'data/images/atlas_256/', 256)
print('Atlas centre cropped and resized to 256x256')
crop_resize('data/raw_images/MClassD/BenchmarkDermoscopic/', 'data/images/MClassD_256/', 256)
print('MClass Dermoscopic centre cropped and resized to 256x256')
crop_resize('data/raw_images/MClassC/BenchmarkClinical/', 'data/images/MClassC_256/', 256)
print('MClass Clinical centre cropped and resized to 256x256')
crop_resize('data/raw_images/ASAN/test-asan test', 'data/images/asan_256/', 256)
print('ASAN centre cropped and resized to 256x256')

# Filtering Fitzpatrick17k dataframe to get names of non-neoplastic lesions
df_fitz = pd.read_csv('data/csv/fitzpatrick17k.csv')
df_fitz = df_fitz.loc[(df_fitz.three_partition_label != 'non-neoplastic') & (df_fitz.qc != '3 Wrongly labelled'), :]
df_fitz = df_fitz.loc[df_fitz['url'].str.contains('http', na=False), :]
df_fitz = df_fitz.loc[df_fitz['fitzpatrick'] != -1, :]
df_fitz = pd.get_dummies(df_fitz, columns=['three_partition_label'], drop_first=True)
df_fitz.rename(columns={'three_partition_label_malignant': 'target'}, inplace=True)

# Downloading raw Fitzpatrick17k images
df_fitz['image_name'] = 0
counter = 0
for i, url in enumerate(df_fitz.url):
    if 'atlasderm' in url:
        path = f"./data/raw_images/fitzpatrick17k/atlas{i}.jpg"
        df_fitz.loc[df_fitz['url'] == url, 'image_name'] = f'atlas{i}.jpg'
    else:
        path = f"./data/raw_images/fitzpatrick17k/{url.split('/', -1)[-1]}"
        df_fitz.loc[df_fitz['url'] == url, 'image_name'] = url.split('/', -1)[-1]
    try:
        urllib.request.urlretrieve(url, path.rstrip('\n'))
    except Exception:
        continue
print('Raw Fitzpatrick17k images downloaded')

# Centre cropping and resizing Fitzpatrick17k to 256x256
crop_resize('./data/raw_images/fitzpatrick17k/', 'data/images/fitzpatrick17k_256/', 256)
print('Fitzpatrick17k centre cropped and resized to 256x256')

# Deleting temporary store of raw images
shutil.rmtree('data/raw_images/')
print('Temporary raw images discarded')
