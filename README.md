# Detecting Melanoma Fairly: Skin Tone Detection and Debiasing for Skin Lesion Classification [[PDF](https://arxiv.org/pdf/2202.02832.pdf)]

## Method:

"Convolutional Neural Networks have demonstrated dermatologist-level performance in the classification of melanoma and other skin lesions, but performance disparities between differing skin tones is an issue that should be addressed before widespread deployment. In this work, we look to uncover and subsequently tackle skin tone bias in melanoma classification. We utilise a modified variational autoencoder [[1]](https://www.aies-conference.com/2019/wp-content/papers/main/AIES-19_paper_220.pdf) to uncover skin tone bias in datasets commonly used as benchmarks. We propose a simple algorithm for automatically labelling the skin tone of lesion images, and use this to annotate the benchmark ISIC dataset. We subsequently use two leading bias unlearning techniques [[2]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Learning_Not_to_Learn_Training_Deep_Neural_Networks_With_Biased_CVPR_2019_paper.pdf) 
[[3]](https://www.robots.ox.ac.uk/~vgg/publications/2018/Alvi18/alvi18.pdf) to mitigate skin tone bias. Our experimental results provide evidence that our skin tone detection algorithm outperforms existing solutions and that unlearning skin tone improves generalisation and can reduce the performance disparity between lighter and darker skin tones [[4]](https://arxiv.org/abs/2104.09957)."

[[Bevan and Atapour-Abarghouei, 2021](https://arxiv.org/abs/2202.02832)]

---
---

## Usage

### Software used (see `requirements.txt` for package requirements)

Python 3.9.6

CUDA Version 11.3

Nvidia Driver Version: 465.31

PyTorch 1.8.1

---

### Downloading data

A free account must be created to download The Interactive Atlas of Dermoscopy, available at this link:
[https://derm.cs.sfu.ca/Download.html](https://derm.cs.sfu.ca/Download.html). Place the `release_v0.zip` file into the
`data/raw_images` directory (see below), from which it will be processed by the `download.py` script. The other datasets
will be automatically downloaded and processed by the `download.py` script.

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

Run `download.py` to download, crop and resize the ISIC, ASAN, MClass clinical, MClass dermoscopic and Fitzpatrick17k
datasaets. Have patience as it may take around an hour to complete. The 256x256 resized images are automatically placed
into `data/images` as shown below. The manually downloaded Atlas data (`data/raw_images/release_v0.zip`) will also be
processed by this script. Note this script clears the `data/images` directory before populating it, so if you want to put other
images in there, do this after running the `download.py` script.

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

---

### Training and evaluation

Training commands for the main experiments from the paper are below. Please see `arguments.py` for the full range of arguments if you wish to devise alternative experiments. Test results (plots, logs and weights) will autosave into the `results` directory, in subdirectories specific to the test number. Please contact me if you require trained weights for any model in the report.

Some useful arguments to tweek the below commands:
* Adjust `--CUDA_VISIBLE_DEVICES` and `num-workers` to suit the available GPUs and CPU cores respectively on your machine.
* To run in debug mode add `--DEBUG` (limits epochs to 3 batches)
* To rerun with different random seeds, use `--seed` argument
* To run on different architechtures, use `--arch` argument to choose from `resnext101`, `enet`, `resnet101`, `densenet` or `inception` (default=`resnext101`)
* Add `--cv` to perform cross validation
* Add `--test-only` if you wish to load weights and run testing only (loads weights of whatever `--test-no` argument is passed).

***Unlearning skin type when training on Fitzpatrick17k data (trains on types 1&2 skin and tests on types 3&4, and 5&6 skin)***
<pre>
<b>Baseline:</b> python train.py --test-no 13 --n-epochs 30 --sktone --CUDA_VISIBLE_DEVICES 0,1 --dataset Fitzpatrick17k --split-skin-types
<b>LNTL:</b> python train.py --test-no 14 --n-epochs 30 --debias-config LNTL --GRL --sktone --CUDA_VISIBLE_DEVICES 0,1 --dataset Fitzpatrick17k --split-skin-types
<b>TABE:</b> python train.py --test-no 15 --n-epochs 30 --debias-config TABE --sktone --CUDA_VISIBLE_DEVICES 0,1 --dataset Fitzpatrick17k --split-skin-types
<b>CLGR:</b> python train.py --test-no 16 --n-epochs 30 --debias-config TABE --GRL --sktone --CUDA_VISIBLE_DEVICES 0,1 --dataset Fitzpatrick17k --split-skin-types
</pre>

Pre labelled skin types are provided in `data/csv/isic_train_20-19-18-17.csv`, but if you wish to run the labelling algorithm, run `preprocessing.py`.

***Unlearning skin type when training on full ISIC data***
<pre>
<b>Baseline:</b> python train.py --test-no 17 --n-epochs 4 --sktone --CUDA_VISIBLE_DEVICES 0,1
<b>LNTL:</b> python train.py --test-no 18 --n-epochs 4 --debias-config LNTL --GRL --sktone --CUDA_VISIBLE_DEVICES 0,1 --num-aux 6
<b>TABE:</b> python train.py --test-no 19 --n-epochs 4 --debias-config TABE --sktone --CUDA_VISIBLE_DEVICES 0,1 --num-aux 6
<b>CLGR:</b> python train.py --test-no 20 --n-epochs 4 --debias-config TABE --GRL --sktone --CUDA_VISIBLE_DEVICES 0,1 --num-aux 6
</pre>

---

### Debiasing Variational Autoencoder

We also use a [debiasing variational autoencoder](https://www.aies-conference.com/2019/wp-content/papers/main/AIES-19_paper_220.pdf) to uncover skin type bias in the ISIC dataset, adapted from JMitnik et al.'s [implementation](https://github.com/JMitnik/FacialDebiasing). The code for this is stored in the `DB_VAE` directory, except for the main script `run_db_vae.py` which is stored in the root directory.

To train and evaluate this run a variation the following command:
<pre>
python run_db_vae.py  --test-no 28 --epochs 150 --DP --z-dim 512 --debias-type max50
</pre>

To perturb each of the top 50 latent variables in turn, add the below to the command (<b>x1</b> to be replace by the source image index and <b>x2</b> to be replaced by the target image index):
<pre>
python run_db_vae.py  --test-no 28 --epochs 150 --DP --z-dim 512 --debias-type max50 --run-mode perturb --var-to-perturb 50 --interp1 <b>x1</b> --interp2 <b>x2</b> --load-model
</pre>

To perturb a specific identified latent variable of the 50, make the command as below (<b>v1</b> being the index of the identified latent variable)
<pre>
python run_db_vae.py  --test-no 28 --epochs 150 --DP --z-dim 512 --debias-type max50 --run-mode perturb --var-to-perturb <b>v1</b> --interp1 <b>x1</b> --interp2 <b>x2</b> --load-model --perturb-single
</pre>
---

## Reference:

[Detecting Melanoma Fairly: Skin Tone Detection and Debiasing for Skin Lesion Classification](https://arxiv.org/abs/2202.02832)
(P. Bevan, A. Atapour-Abarghouei) [[pdf](https://arxiv.org/pdf/2202.02832.pdf)]

```
@misc{bevan2022detecting,
      title={Detecting Melanoma Fairly: Skin Tone Detection and Debiasing for Skin Lesion Classification}, 
      author={Peter J. Bevan and Amir Atapour-Abarghouei},
      year={2022},
      eprint={2202.02832},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
---
