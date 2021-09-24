import os
import pandas as pd


# ASAN
def create_asan_csv():
    asan_names = []

    for subdir, dir, files in os.walk('D:/2.Data/ASAN/asan-test/biopsy/'):
        for file in files:
            asan_names.append(file)

    df = pd.DataFrame(asan_names, columns=['image_name'])

    df['diagnosis'] = None

    for i, im in enumerate(df.image_name):
        if im in os.listdir('D:/2.Data/ASAN/asan-test/biopsy/actinickeratosis/'):
            df.iloc[i, 1] = 'AK'
        if im in os.listdir('D:/2.Data/ASAN/asan-test/biopsy/basalcellcarcinoma/'):
            df.iloc[i, 1] = 'BCC'
        if im in os.listdir('D:/2.Data/ASAN/asan-test/biopsy/dermatofibroma/'):
            df.iloc[i, 1] = 'DF'
        if im in os.listdir('D:/2.Data/ASAN/asan-test/biopsy/lentigo/'):
            df.iloc[i, 1] = 'BKL'
        if im in os.listdir('D:/2.Data/ASAN/asan-test/biopsy/actinickeratosis/'):
            df.iloc[i, 1] = 'AK'
        if im in os.listdir('D:/2.Data/ASAN/asan-test/biopsy/malignantmelanoma/'):
            df.iloc[i, 1] = 'melanoma'
        if im in os.listdir('D:/2.Data/ASAN/asan-test/biopsy/pigmentednevus/'):
            df.iloc[i, 1] = 'nevus'
        if im in os.listdir('D:/2.Data/ASAN/asan-test/biopsy/seborrheickeratosis/'):
            df.iloc[i, 1] = 'BKL'
        if im in os.listdir('D:/2.Data/ASAN/asan-test/biopsy/squamouscellcarcinoma/'):
            df.iloc[i, 1] = 'SCC'

    df.to_csv('data/csv/asan.csv', index=False)


# MClass
def create_mclass_csv():
    mclassd_names = []

    for subdir, dir, files in os.walk('data/raw_images/MClassD/BenchmarkDermoscopic/benign'):
        for file in files:
            mclassd_names.append(file)

    df_benign = pd.DataFrame(mclassd_names, columns=['image_name'])
    df_benign['target'] = 0

    mclassd_names = []

    for subdir, dir, files in os.walk('data/raw_images/MClassD/BenchmarkDermoscopic/malign'):
        for file in files:
            mclassd_names.append(file)

    df_malignant = pd.DataFrame(mclassd_names, columns=['image_name'])
    df_malignant['target'] = 1

    df = pd.concat([df_benign, df_malignant])

    df.to_csv('csv/MClassD.csv', index=False)

    # MClassC

    mclassc_names = []

    for subdir, dir, files in os.walk('data/raw_images/MClassC/BenchmarkClinical/benign'):
        for file in files:
            mclassc_names.append(file)

    df_benign = pd.DataFrame(mclassc_names, columns=['image_name'])
    df_benign['target'] = 0

    mclassc_names = []

    for subdir, dir, files in os.walk('data/raw_images/MClassC/BenchmarkClinical/malignant'):
        for file in files:
            mclassc_names.append(file)

    df_malignant = pd.DataFrame(mclassc_names, columns=['image_name'])
    df_malignant['target'] = 1

    df = pd.concat([df_benign, df_malignant])

    df.to_csv('csv/MClassC.csv', index=False)


def remove_mclass_ISIC():
    # Flagging items present in mclassd in ISIC to remove for data leakage
    df_isic = pd.read_csv('csv/isic_train_20-19-18-17.csv')
    df_mclassd = pd.read_csv('csv/MClassD.csv')
    mclassd_lst = df_mclassd.image_name.apply(lambda x: x[:-4])  # List of mclassd images
    df_isic['mclassd'] = 0
    df_isic.loc[df_isic.image_name.isin(mclassd_lst), 'mclassd'] = 1  # Marking in isic df if in mclassd

    df_isic.to_csv('csv/isic_train_20-19-18-17.csv', index=False)  # Resaving isic csv
