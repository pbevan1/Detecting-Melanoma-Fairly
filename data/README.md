### Downloading data

A free account must be created to download The Interactive Atlas of Dermoscopy, available at this link: [https://derm.cs.sfu.ca/Download.html](https://derm.cs.sfu.ca/Download.html). Place the `release_v0.zip` file into the `data/raw_images` directory (see below), from which it will be processed by the `download.py` script. The other datasets will be automatically downloaded and processed by the `download.py` script.

<pre>
Melanoma-Bias  
└───Data
│   └───csv
│   |   │   asan.csv
│   |   │   atlas.csv
│   |   │   ...
│   |
|   └───images
|   |
|   └───raw_images
|       |   <b>release_v0.zip</b>
|
...
</pre>

Run `download.py` to download, crop and resize the ISIC, ASAN, MClass clinical, MClass dermoscopic and Fitzpatrick17k datasaets. Have patience as it may take an hour or so to complete. The 256x256 resized images are automatically placed into `data/images` as shown below. The manually downloaded Atlas data (`data/raw_images/release_v0.zip`) will also be processed by this script.

**NOTE**: The surgical markings/rulers test set from Heidelberg University [[3]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6694463/) is not publicly available. Refer to the paper for contact details of the corresponding author to request data if necessary.

The data directory should now look as follows:
<pre>
Melanoma-Bias  
└───Data
│   └───csv
│   |   │   asan.csv
│   |   │   atlas.csv
│   |   │   ...
│   |
|   └───images
|       |   asan_256
|       |   atlas_256
|       |   fitzpatrick17k_256
|       |   isic_19_train_256
|       |   isic_20_train_256
|       |   MClassC_256
|       |   MClassD_256
|
...
</pre>

If you do wish to manually download the datasets, they are available at the following links:

ISIC 2020 data: [https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256](https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256)  
ISIC 2019/2018/2017 data: [https://www.kaggle.com/cdeotte/jpeg-isic2019-256x256](https://www.kaggle.com/cdeotte/jpeg-isic2019-256x256)  
Interactive Atlas of Dermoscopy: [https://derm.cs.sfu.ca/Welcome.html](https://derm.cs.sfu.ca/Welcome.html)  
ASAN Test set: [https://figshare.com/articles/code/Caffemodel_files_and_Python_Examples/5406223](https://figshare.com/articles/code/Caffemodel_files_and_Python_Examples/5406223)  
MClassC/MClassD: [https://skinclass.de/mclass/](https://skinclass.de/mclass/)