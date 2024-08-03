from models.unet3d_latent_888 import UNet3D_Latent_888
from models.vqvae import VQVAE
from models.enc3d_vqvae import Encoder3D_VQVAE
from models.dec3d_vqvae import Decoder3D_VQVAE

def create_model(config):
    if config["model_name"] == "vqvae":
        model = VQVAE(
            n_elements=len(config["elements"]), 
            n_latent=config["n_latent"], 
            n_channels=config["n_channels"],
            ch_mults=config["ch_mults"],
            is_attn=config["is_attn"],
            n_blocks=config["n_blocks"],
            n_groups=config["n_groups"],
            dropout=config["dropout"],
            smooth_sigma=config["smooth_sigma"],
            embedding_dim=config["embedding_dim"],
            num_embeddings=config["num_embeddings"],
            commitment_cost=config["commitment_cost"],
            skip_connections=config["skip_connections"],
        )
   
    elif config["model_name"] == "unet3d_latent_vqvae_888":
        model = UNet3D_Latent_888(
            n_latent=config["n_latent"], 
            n_channels=config["n_channels_latent"],
            vqvae_out_dim=config["vqvae_out_dim"],
            ch_mults=config["ch_mults_latent"],
            is_attn=config["is_attn"],
            n_blocks=config["n_blocks"],
            n_groups=config["n_groups_latent"],
            dropout=config["dropout"],
            smooth_sigma=config["smooth_sigma"],
            skip_connections=config["skip_connections_latent"]
        )
    
    else:
        NotImplementedError(f"{config['model_type']} Not implemented yet")

    return model

def create_encoder_model(config):
    
    if config["model_outer_encoder"] == "encoder3d_vqvae":
        model_outer_encoder = Encoder3D_VQVAE(
            n_elements=len(config["elements"]),
            n_latent=config["n_latent"],
            n_channels=config["n_channels"],
            ch_mults=config["ch_mults"],
            is_attn=config["is_attn"],
            n_blocks=config["n_blocks"],
            n_groups=config["n_groups"],
            dropout=config["dropout"],
            embedding_dim=config["embedding_dim"],
            num_embeddings=config["num_embeddings"],
            commitment_cost=config["embedding_dim"],

        )
    return model_outer_encoder

def create_decoder_model(config):

    if config["model_outer_decoder"] == "decoder3d_vqvae":
        model_outer_decoder = Decoder3D_VQVAE(
            n_elements=len(config["elements"]),
            n_channels=config["n_channels"],
            ch_mults=config["ch_mults"],
            is_attn=config["is_attn"],
            n_blocks=config["n_blocks"],
            n_groups=config["n_groups"],
            dropout=config["dropout"],
            skip_connections=config["skip_connections"],
            embedding_dim=config["embedding_dim"]
        )
    return model_outer_decoder

