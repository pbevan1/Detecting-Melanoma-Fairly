### Debiasing Variational Autoencoder

We also use a debiasing variational autoencoder (DB-VAE) [[5]](https://www.aies-conference.com/2019/wp-content/papers/main/AIES-19_paper_220.pdf) to uncover skin type bias in the ISIC dataset, adapted from JMitnik et al.'s [implementation](https://github.com/JMitnik/FacialDebiasing). The main script `run_db_vae.py` is stored in the root directory, this should be used to run the model.

To train and evaluate this run a variation the following command:
<pre>
python run_db_vae.py  --test-no 28 --epochs 50 --DP --z-dim 512 --debias-type max50
</pre>

To perturb each of the top 50 latent variables in turn, add the below to the command (<b>x1</b> to be replace by the source image index and <b>x2</b> to be replaced by the target image index):
<pre>
python run_db_vae.py  --test-no 28 --epochs 50 --DP --z-dim 512 --debias-type max50 --run-mode perturb --var-to-perturb 50 --interp1 <b>x1</b> --interp2 <b>x2</b> --load-model
</pre>

To perturb a specific identified latent variable of the 50, make the command as below (<b>v1</b> being the index of the identified latent variable)
<pre>
python run_db_vae.py  --test-no 28 --epochs 50 --DP --z-dim 512 --debias-type max50 --run-mode perturb --var-to-perturb <b>v1</b> --interp1 <b>x1</b> --interp2 <b>x2</b> --load-model --perturb-single
</pre>

