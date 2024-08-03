import argparse
import yaml
from dataclasses import dataclass, field

def parse_args():
    parser = argparse.ArgumentParser("voxmol", add_help=False)
    parser.add_argument(
        "--debug", action="store_true", help="debug mode"
    )
    parser.add_argument(
        "--device", default="cuda", help="device"
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="seed"
    )
    parser.add_argument(
        "--num_workers", default=2, type=int, help="num workers"
    )
    parser.add_argument(
        "--exp_dir", default="output/", type=str, help="experiment dir"
    )
    parser.add_argument(
        "--exp_name", default=None, type=str, help="experiment name."
    )
    parser.add_argument(
        "--wandb", default=0, type=int, help="use wandb if > 0"
    )

    # data args
    parser.add_argument(
        "--data_dir", default="dataset/data/", type=str, help="dataset path" #on SHPC
    )
    parser.add_argument(
        "--dset_name", default="drugs", type=str, help="dataset name"
    )
    parser.add_argument(
        "--rot_aug", default=1, type=int, help="augment training data w rotations if >1"
    )
    parser.add_argument(
        "--trans_aug", default=0.5, type=int, help="augment training translations if >1"
    )
    parser.add_argument(
        "--smooth_sigma", default=1.8, type=float, help="varaince for smooth density "
        + "for WJS training"
    )
    parser.add_argument(
        "--resolution", default=0.25, type=float, help="resolution of each voxel"
    )
    parser.add_argument(
        "--grid_dim", default=None, type=int, help="value for each dimension "
        + "of voxel grid"
    )
    parser.add_argument(
        "--grid_dim_latent", default=None, type=int, help="value for each latent dimension "
        + "of voxel grid"
    )
    parser.add_argument(
        "--fix_radius", default=0.5, type=float, help="use same fix radius for all "
        + "atoms if not None"
    )
    parser.add_argument(
        "--elements", default=None, type=str, help="string w elements separated by"
        + " comma. If None, use all elements for each dataset"
    )

    # training args
    parser.add_argument(
        "--num_epochs", default=150, type=int, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="batch size for training"
    )
    parser.add_argument(
        "--amp", default=0, type=int, help="use autoamtic mixed precision"
    )

    # optim args
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="absolute learning rate"
    )
    parser.add_argument(
        "--wd", type=float, default=1e-2, help="weight decay coeff."
    )
    parser.add_argument(
        "--scheduler", default=0, type=int, help="use scheduler"
    )

    # model args
    parser.add_argument(
        "--model_config", default="models/configs/latent_config.yml", help="model path"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="dropout on attention layers"
    )
    parser.add_argument(
        "--model_ema", type=int, default=1, help="use exponential moving average for weight updates"
    )

    # wjs args
    parser.add_argument(
        "--n_chains", default=10, type=int, help="number of chains to be generated in"  # TODO change to 100  /1
        + "parallel"
    )
    parser.add_argument(
        "--delta", default=.25, type=float, help="delta wjs hp"
    )
    parser.add_argument(
        "--friction", default=1., type=float, help="friction wjs hp"
    )
    parser.add_argument(
        "--lipschitz", default=1., type=float, help="lipschitz wjs hp"
    )
    parser.add_argument(
        "--steps_wjs", default=5, type=int, help="number of walk steps for the wjs"  # TODO change to 100 / 10
    )
    parser.add_argument(
        "--repeats", default=1, type=int, help="number of (batched) wjs chains"
    )
    parser.add_argument(
        "--total_molecules", default=4, type=int, help="number of (batched) wjs chains"
    )
    parser.add_argument(
        "--pretrained_path", default=None, type=str, help="pre-trained model dir"
    )
    parser.add_argument(
        "--pretrained_path_latent", default=None, type=str, help="pre-trained model dir"
    )
    parser.add_argument(
        "--skip_connections", default=0, type=int, help="use skip connections in U-Net"
    )
    parser.add_argument(
        "--skip_connections_latent", default=1, type=int, help="use skip connections in latent U-Net"
    )
    parser.add_argument(
        "--n_latent", default=1024, type=int, help="latent embeding dimension"
    )
    parser.add_argument(
        "--refine", default=True, type=bool, help="refine samples"
    )
    parser.add_argument(
        "--embedding_dim", default=1024, type=int, help="embedding dimension for VQVAE"
    )
    parser.add_argument(
        "--num_embeddings", default=256, type=int, help="number embeddings in VQVAE codebook"
    )
    parser.add_argument(
        "--commitment_cost", default=0.5, type=float, help="beta weight for VQVAE"
    )
    parser.add_argument(
        "--seed_dir", default="dataset/data/seed_data/", type=str, help="project_seed_dir"
    )
    parser.add_argument(
        "--visualize_voxels", default=0, type=bool, help="visualize voxels"
    )
    parser.add_argument(
        "--visualize_smiles", default=0, type=bool, help="visualize smiles"
    )
    parser.add_argument(
        "--input_smiles", default="Brc1c(CSc2nc3ccccc3s2)nc2ncccn12", type=str, help="seed lead SMILES"
    )
    

    args = parser.parse_args()
    config = args.__dict__

    # update global config with model config options
    with open(config["model_config"], "r") as f:
        config_model = yaml.safe_load(f)
    for k, v in config_model.items():
        if k not in config:
            config[k] = v
        elif config[k] is None:
            config[k] = v

    if config["elements"] is not None:
        config["elements"] = config["elements"].split(",")

    #if config["dset_name"] == "drugs":
    if config["elements"] is None:
        config["elements"] = ["C", "H", "O", "N", "F", "S", "Cl", "Br"]
    config["grid_dim"] = 64 if config["grid_dim"] is None else config["grid_dim"]
    config["grid_dim_latent"] = 8 if config["grid_dim_latent"] is None else config["grid_dim_latent"]   
    
    config["num_channels"] = len(config["elements"])
    if config["n_latent"] is None:
        config["n_latent"] = len(config["elements"])

    print(">> config:")
    for k, v in config.items():
        print(f"  | {k}: {v}")

    return config
