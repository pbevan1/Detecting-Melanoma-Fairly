### misc_code
This directory contains supplementary scripts that were used in the project, but which aren't necessary to run the model.

A brief description of the function of each script:

`create_csv.py`: This script was used to create csv files for datasets which were images only, and translates the
storage structure of the image repositories into csv files compatible with the dataloader. Additionally there's a
function that flags images in the ISIC data that are also present in the MClassD data, allowing these to be removed.

`df_random_seeds.py`: This script was used to compile results across the 6 random seeds used in the artefact bias
experiments and save these in a csv format, allowing easy plotting using ggplot in R.

`ggplots.r`: This script was used to make plots for the report using the datasets and results. R was used for it's
intuitive ggplot library.

`marking_detection.py`: This script was used to identify images with surgical markings in the ISIC training data.
Since the method isn't perfect, the images were also manually labelled, but these automated labels were used to help
double check that no surgical markings had been missed.

`ROC_plots.py`: This script was used to create cross-model ROC comparison plots using saved metrics.
