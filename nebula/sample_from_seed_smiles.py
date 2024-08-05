from options import parse_args

import os
import torch

import numpy as np
import torch.backends.cudnn as cudnn

from rdkit import Chem

from dataset.dataset import DatasetVoxMol
from models import create_model, create_encoder_model, create_decoder_model
from voxelizer import Voxelizer
from utils.utils import load_checkpoint, makedir, mol2xyz, create_exp_dir
from utils.midi_utils import process_obabel, open_babel_preprocess
from utils.utils_vis import render_generated_samples
from dataset.create_data import smiles2sdf, preprocess_dataset


def main(config):

    """
    python sample_from_seed_smiles.py \
    --exp_name sample/ \
    --delta 0.25 \
    --n_chains 10 \
    --total_molecules 10 \
    --visualize_smiles 1 \
    --input_smiles "N#Cc1ccc(N=C(O)C(F)(F)Cl)cn1"
    """

    wjs_steps_total = [5, 10, 20, 50, 100]

    # convert the input 1-D SMILES to 3D conformers and pre-process them
    makedir(config["seed_dir"])
    pth_file = os.path.join(config["seed_dir"], "seed.pth")
    sdf_file = os.path.join(config["seed_dir"], "seed.sdf")
    
    smiles2sdf(config["input_smiles"], sdf_file) # saves 3D conformers in an sdf file
    # TODO default 50 conformers generated per seed SMILES
    preprocess_dataset(config["seed_dir"]) # saves pre-processed seeds from sdf in a pth file

    model_outer_encoder, model_outer_decoder, model_inner_ae = load_models(config)

    voxelizer_discrete = Voxelizer(
        grid_dim=config["grid_dim"],
        num_channels=len(config["elements"]),
        resolution=config["resolution"],
        device=config["device"],
    )

    dset = DatasetVoxMol(
        dset_name="",
        data_dir=pth_file, #config["seed_dir"], 
        elements=["C", "H", "O", "N", "F", "S", "Cl", "Br"],
        split="test",
        rotate=False,
        shift=0,
        small=False,
        atomic_radius=.5,
        max_n_atoms=80,
        multi_file=True,  
    )

    count = 0

    create_exp_dir(config)
    dirname_xyz = os.path.join(config['output_dir'], "xyzs")
    makedir(dirname_xyz)

    for ii in range(1): # len(dset)):  # for multiple conformers
        if ii > 0:
            mols_total = []

        for rep in range(config["repeats"]):
            if count >= config["total_molecules"]:
                break
            y, v = initialize_with_seed(config, dset[ii], model_outer_encoder, voxelizer_discrete)
            mols, count = sample(config, y, v, wjs_steps_total, count, voxelizer_discrete, model_inner_ae, model_outer_encoder, model_outer_decoder)

        if ii > 0:
            mols_total.append(mols)

    if ii > 0:
        mols = mols_total

    # convert generated xyz to sdf
    process_obabel(os.path.join(config['output_dir']))
    generated_smiles = open_babel_preprocess(os.path.join(os.getcwd(), config["output_dir"], "molecules_obabel.sdf"), "geom_with_h")
    
    # optionally save the generated SMILES as png
    if config["visualize_smiles"]:
        from rdkit.Chem import Draw

        dirname_smiles = os.path.join(config['output_dir'], "smiles_generated")
        makedir(dirname_smiles)

        smiles_seed_mol1 = Chem.MolFromSmiles(config["input_smiles"])
        smiles_img_seed1 = Draw.MolToImage(smiles_seed_mol1)
        smiles_img_seed1.save(os.path.join(dirname_smiles, "smiles_in_1.png")) 

        for i in range(len(generated_smiles)):
            smiles_seed_mol2 = Chem.MolFromSmiles(generated_smiles[i])
            smiles_img_seed2 = Draw.MolToImage(smiles_seed_mol2)
            smiles_img_seed2.save(os.path.join(dirname_smiles, "smiles_out_1_" + str(i) + ".png"))

    return generated_smiles

def sample(config, y, v, wjs_steps_total, count, voxelizer_discrete, model_inner_ae, model_outer_encoder, model_outer_decoder):
    wjs_steps = [wjs_steps_total[0]]
    if len(wjs_steps_total) > 1:
        wjs_steps.extend(np.diff(np.array(wjs_steps_total)).tolist())
    
    mols_total = []
    step_count = -1
    for wjs_step in wjs_steps:
        step_count += 1
        with torch.no_grad():
            config["steps_wjs"] = wjs_step
            y, v = model_inner_ae.wjs_walk_steps(y, v, config)  # walk steps
            xhats_latent = model_inner_ae.wjs_jump_step(y)  # jump

            xhats_latent = normalize_latents(config, xhats_latent, norm= "unnorm")

            # add quantization again before passing the denoised latents to the decoder
            _, xhats_latent, _, _, _ = model_outer_encoder._vq_vae(xhats_latent)
            xhats = model_outer_decoder(xhats_latent, None)
            xhats[xhats < .2] = 0

            mols = voxelizer_discrete.vox2mol(xhats, refine=True)

            if config["visualize_voxels"]:
                dirname_vox = os.path.join(config['output_dir'], "voxels_generated")
                makedir(dirname_vox)
                num_voxels_to_save = min(xhats.shape[0], 10)  
                render_generated_samples(xhats[0:num_voxels_to_save, :, :, :,:].detach().cpu(), dirname_vox)  

        if mols is not None:
            for i in range(xhats.shape[0]):
                try:
                    mol = mols[i]
                    xyz_str = mol2xyz(mol)

                    with open(os.path.join(os.path.join(config['output_dir'], "xyzs"), f"sample_{count:05d}_step{wjs_steps_total[step_count]:05d}.xyz"), "w") as f:
                        f.write(xyz_str)
                    f.close()
                    count += 1

                except Exception:
                    print(">> molecule not valid")
                    continue
        mols_total.append(mols)

    return mols_total, count


def load_models(config):

    pretrained_path = "output/vqvae/checkpoint.pth.tar"
    pretrained_path_latent = "output/latent/checkpoint.pth.tar"

    model_outer_encoder = create_encoder_model(config)
    model_outer_decoder = create_decoder_model(config)
    model_inner_ae = create_model(config)
    
    model_outer_encoder.to(config["device"]) 
    model_outer_decoder.to(config["device"]) 
    model_inner_ae.to(config["device"]) 

    model_outer_encoder, epoch_compr = load_checkpoint(model_outer_encoder, pretrained_path)
    model_outer_decoder, epoch_compr = load_checkpoint(model_outer_decoder, pretrained_path)
    model_inner_ae, epoch_latent = load_checkpoint(model_inner_ae, pretrained_path_latent)

    model_outer_encoder.eval()
    model_outer_decoder.eval()
    model_inner_ae.eval()

    return model_outer_encoder, model_outer_decoder, model_inner_ae


def initialize_with_seed(config, sample, model_outer_encoder, voxelizer_discrete):
    sample["coords"] = sample["coords"].unsqueeze(0)
    sample["atoms_channel"] = sample["atoms_channel"].unsqueeze(0)
    sample["radius"] = sample["radius"].unsqueeze(0)

    seed = voxelizer_discrete(sample)

    # compress the seed lead compound to latent embeddings
    y, _, _ = model_outer_encoder(seed.repeat(config["n_chains"],1,1,1,1))
    v = torch.zeros_like(y)

    # normalize latent embeddings to 0-mean, 1-std
    y = normalize_latents(config, y, norm= "norm")

    # add noise to the seed
    y = y + torch.cuda.FloatTensor(y.shape).normal_(0, config["smooth_sigma"])

    return y, v

def normalize_latents(config, y, norm= "norm"):

    latent_min = torch.from_numpy(np.load("output/vqvae/channel_latent_min.npy")).to(config["device"]) 
    latent_max = torch.from_numpy(np.load("output/vqvae/channel_latent_max.npy")).to(config["device"]) 
    latent_mean = torch.from_numpy(np.load("output/vqvae/channel_latent_mean.npy")).to(config["device"])
    latent_std = torch.from_numpy(np.load("output/vqvae/channel_latent_std.npy")).to(config["device"])

    if norm == "norm":
        y_shape = y.shape
        y = y.view(y.shape[0], -1)
        y = (y-latent_mean)/latent_std
        y = y.view(y_shape)
    elif norm == "unnorm":
        y_shape = y.shape
        y = y.view(y.shape[0], -1)
        y = y * latent_std + latent_mean
        y = y.view(y_shape)

    return y


if __name__ == "__main__":
    config = parse_args()
    seed = config["seed"]

    print(">> n gpus available:", torch.cuda.device_count())
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True

    main(config)
