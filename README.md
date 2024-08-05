# NEBULA 💫
**N**eural **E**mpirical **B**ayes **U**nder **La**tent Representations for Efficient and Controllable Design of Molecular Libraries


## Temporary repo for legal review before public code release

This repository contains the implementation of `NEBULA` presented in AI4Science Workshop at ICML 2024 titled [Neural Empirical Bayes Under Latent Representations for Efficient and Controllable Design of Molecular Libraries]().

If you use the code, please cite this paper.
```bibtex

@article{nebula_2024,
  title={NEBULA: Neural Empirical Bayes Under LAtent Representations for Efficient and Controllable Design of Molecular Libraries},
  author={Nowara, Ewa M and Pinheiro, Pedro O and Mahajan, S Pooja and Mahmood, Omar and Watkins, Andrew M and and Saremi, Saeed and Maser Michael},
  journal={ICML 2024 AI for Science workshop},
  year={2024}
}

```


<details open><summary><b>Table of contents</b></summary>

- [What is NEBULA](#why-use)
- [Install Instructions](#install)
- [Data Preparation](#data)
- [Training](#train)
- [Generation](#sample)
- [License](#license)
</details>


## A latent generative model for fast seeded genreration of small molecules <a name="why-use"></a>

`nebula` is a latent 3D generative model for scalable generation of large molecular libraries around a seed compound of interest. Led by [Ewa Nowara](https://www.gene.com/scientists/our-scientists/ewa-nowara), [Pedro O. Pinheiro](https://www.gene.com/scientists/our-scientists/pedro-o-pinheiro), [Sai Pooja Mahajan](https://www.linkedin.com/in/sai-pooja-mahajan-88272910), [Omar Mahmood](https://www.linkedin.com/in/omar-mahmood), [Andrew M. Watkins](https://www.gene.com/scientists/our-scientists/andy-watkins), [Saeed Saremi](https://www.linkedin.com/in/saeed-saremi-71935916), and [Michael Maser](https://gene.com/scientists/our-scientists/michael-maser) at [Prescient Design, Genentech](https://www.gene.com/scientists/our-scientists/prescient-design).

Below we show a example generated samples for seed compounds from within-dataset [GEOM-Drugs](https://www.nature.com/articles/s41597-022-01288-4) (top), cross-dataset generalization to [PubChem](https://arxiv.org/abs/2305.18454) (middle), and cross-dataset generalization to [recent cancer drugs](https://drughunter.com/articles/acs-spring-2024-first-time-disclosures/) (bottom). 

<p align="center">
<img src="figures/NEBULA_generations.jpg" width=20000px>
</p>

* NEBULA generates large molecular libraries around a seed ligand molecule nearly an order of magnitude faster than existing methods without sacrificing sample quality.
* NEBULA generalizes very well to unseen drug-like molecules

## Install <a name="install"></a>
We assume the user has anaconda (or, preferably mamba) installed and has access to GPU. 

Clone the repo, cd into it and do 
```bash
mamba env create -f env.yaml
conda activate nebula
pip install -e .
```


## Prepare data <a name="data"></a>
To pre-process a dataset, run:
```
cd nebula/dataset; python create_data.py
```
This script expects `.sdf` files with confromers of molecules and it outputs the pre-processed data in `.pth` files.

Please refer to [MiDi](https://github.com/cvignac/MiDi) or [VoxMol](https://github.com/Genentech/voxmol) for downloading the GEOM-Drugs dataset and train/validation/test splits.

## Train NEBULA from scratch <a name="train"></a>
We cannot release the pre-trained model weights at this time due to legal reasons. 

To train NEBULA, first train the compression model used to obtain the latent embeddings. Then train the latent denoising model in the learned latent space.

To train the **compression model** on GEOM-Drugs, run inside `nebula/` directory:

```
cd nebula

python train_vqvae.py \
  --model_config models/configs/vqvae_config.yml \
  --exp_name vqvae/ 
```

To train the **latent model** on GEOM-Drugs, run inside `nebula/` directory:

```
cd nebula

python train_latent.py \
  --model_config models/configs/latent_config.yml \
  --pretrained_path output/vqvae/checkpoint.pth.tar \
  --exp_name latent/ 
```

These scripts will train a the models on the train set and evaluate the reconstruction performance on the validation set at each epoch. 

Use the flag `--wandb 1` if you want to log results on wandb. 
`--debug 1` can be used to debug training on a subset of the dataset

See `options.py` to see all argument options and default values.

The GEOM-drugs models were both trained with batch size 32 (on 4 GPUs) for 150 epochs or until the meain intersection over union between the ground truth and reconstructed voxels reaches 0.90.

## Generate samples with pre-trained NEBULA <a name="sample"></a>
Once the compression and latent models have been trained, generate new samples around a seed of interest using this command:

```
cd nebula

python sample_from_seed_file.py \
  --exp_name sample/ \
  --delta 0.25 \
  --total_molecules 2 \
  --visualize_voxels 0 \
  --visualize_smiles 1
```

Alternatively, generation can be performed on a new input sequence of 1D SMILES of interest (the 3D conformer with xyz coordinates will be computationally generated for it):

```
cd nebula

python sample_from_seed_smiles.py \
  --exp_name sample/ \
  --delta 0.25 \
  --total_molecules 2 \
  --input_smiles "Brc1c(CSc2nc3ccccc3s2)nc2ncccn12" \
  --visualize_voxels 0 \
  --visualize_smiles 1
```

It saves the generated molecules as xyz files and it post-processes them to obtain generated SMILES.

`--total_molecules` - total molecules to be generated around a seed after all steps: default: 2
`--n_chains` - number of molecules generated in parallel at once for each sampling step (depends on GPU capacity), default: 1
`--delta` - step size (smaller size will stay closer to the seed and will require more steps to generate new molecules), default: 0.25
`--repeats` - number of times to repeat an experiment to reach the total number of desired molecules (`total_molecules`), default: 1
`--seed_dir` - directory with sdf of a seed of interest to generate a library around, default: `dataset/data/seed_data/`
`--visualize_voxels 1` - optionally visualize generated voxels (very slow, disabled by default)
`--visualize_smiles 1` - optionally visualize generated voxels (disabled by default)


`steps_wjs_total` - list of how many sampling steps to take (e.g., [5, 10] steps means that molecules will be generated after 5 and after 10 steps), it needs to be set in `sample_from_seed_from_file.py` or `sample_from_seed_smiles.py` scripts.


## License <a name="license"></a>
This project is under the Apache license, version 2.0. See LICENSE for details.

