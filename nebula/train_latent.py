import os
import time
import random
import numpy as np
import wandb
import torch
import torchmetrics
import torch.backends.cudnn as cudnn

from options import parse_args

from dataset.dataset import DatasetVoxMol
from voxelizer import Voxelizer

from models import create_model, create_encoder_model, create_decoder_model
from models.ema import ModelEma  
from models.adamw import AdamW   

from metrics import MetricsDenoise
from utils.utils import load_checkpoint, save_checkpoint, create_exp_dir, create_exp_name


def main(config):
    '''
    This script trains a 3D denoising U-Net in the latent space to reconstruct molecules in the latent space, 
    given latent embeddings of the molecules with added noise.

    python train_latent.py --model_config models/configs/latent_config.yml \
    --exp_name latent/ \
    --pretrained_path output/vqvae/checkpoint.pth.tar 

    '''

    dset_train = DatasetVoxMol(
        dset_name=config["dset_name"],
        data_dir=config["data_dir"],
        elements=config["elements"],
        split="train",
        rotate=config["rot_aug"] > 0,
        shift=0,
        small=config["debug"],
        atomic_radius=.5,
        max_n_atoms=800,
        multi_file=False,
    )

    # use a random split to train a larger dataset
    if config["debug"]:
        dset_train = torch.utils.data.Subset(dset_train, range(100))
        dataset_subset = [1] 
        
    else:
        dataset_subset = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  

    generator = torch.Generator().manual_seed(config["seed"])
    dset_train_split = torch.utils.data.random_split(dset_train, dataset_subset, generator=generator)

    loader_train = torch.utils.data.DataLoader(
        dset_train,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True, 
        pin_memory=True, 
        drop_last=True,
    )
    
    dset_val = DatasetVoxMol(
        dset_name=config["dset_name"],
        data_dir=config["data_dir"],
        elements=config["elements"],
        split="val",
        rotate=False,
        shift=0,  
        small=config["debug"],
        atomic_radius=.5,
        max_n_atoms=80,
        multi_file=False,
    )

    if config["debug"]:
        dset_val = torch.utils.data.Subset(dset_val, range(10))

    loader_val = torch.utils.data.DataLoader(
        dset_val,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    dset_val_split = torch.utils.data.random_split(dset_val, dataset_subset, generator=generator)

    print(f">> training/val set sizes: {len(dset_train)}/{len(dset_val)}")

    voxelizer = Voxelizer(
        grid_dim=config["grid_dim"],
        num_channels=len(config["elements"]),
        resolution=config["resolution"],
        device=config["device"],
    )

    model_outer_encoder = create_encoder_model(config)
    model_outer_decoder = create_decoder_model(config)
    model_inner_ae = create_model(config)

    if torch.cuda.device_count() > 1:  
        model_inner_ae = torch.nn.DataParallel(model_inner_ae)

    model_outer_encoder.to(config["device"])
    model_outer_decoder.to(config["device"])
    model_inner_ae.to(config["device"])
    
    n_params1 = sum(p.numel() for p in model_outer_encoder.parameters() if p.requires_grad)
    n_params2 = sum(p.numel() for p in model_outer_decoder.parameters() if p.requires_grad)
    n_params = n_params1 + n_params2
    print(f">> outer AE model has {(n_params/1e6):.02f}M parameters")

    n_params = sum(p.numel() for p in model_inner_ae.parameters() if p.requires_grad)
    print(f">> inner AE model has {(n_params/1e6):.02f}M parameters")

    if config["pretrained_path"]:
        model_outer_encoder, _ = load_checkpoint(model_outer_encoder, config["pretrained_path"])
        model_outer_decoder, _ = load_checkpoint(model_outer_decoder, config["pretrained_path"])
    
    else:
        print(">> There is no pre-trained compression model")
    
    # freeze weights of the compression model
    for name, para in model_outer_encoder.named_parameters():
        para.requires_grad = False

    for name, para in model_outer_decoder.named_parameters():
        para.requires_grad = False

    if config["pretrained_path_latent"]:
        model_inner_ae, epoch_start = load_checkpoint(model_inner_ae, config["pretrained_path_latent"])
    else:
        print(">> Model is running with random initialization...")
        epoch_start = 0

    criterion = torch.nn.MSELoss().to(config["device"])

    optimizer = AdamW(  
        model_inner_ae.parameters(),
        lr=config["lr"],
        weight_decay=config["wd"],
        betas=[0.99, 0.999],
    )

    optimizer.zero_grad() 

    if config["scheduler"] == 1:
        scheduler = CosineAnnealingLR(optimizer, T_max=len(dset_train) * config["num_epochs"])
    elif config["scheduler"] == 2:
        scheduler = StepLR(optimizer, step_size=160, gamma=0.1)
    else:
        scheduler = None

    scaler = None  
    if config["amp"] > 0:
        print(">> train with automtic mixed precision (AMP)")
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"] > 0)

    model_inner_ae_ema = ModelEma(model_inner_ae, decay=.998)

    metrics = MetricsDenoise(
        loss=torchmetrics.MeanMetric(),
        miou=torchmetrics.classification.BinaryJaccardIndex(),
        miou_latent=torchmetrics.classification.BinaryJaccardIndex(),
    )
    metrics.to(config["device"])

    best_miou = 0

    print(">> start training...")
    latent_mean, latent_std, latent_min, latent_max = compute_latent_stats(dset_train, model_outer_encoder, config, voxelizer)
    
    latent_mean = torch.from_numpy(latent_mean).to(config["device"])
    latent_std = torch.from_numpy(latent_std).to(config["device"])

    latent_min = torch.from_numpy(latent_min).to(config["device"])
    latent_max = torch.from_numpy(latent_max).to(config["device"])

    for epoch in range(epoch_start, config["num_epochs"]):
        t0 = time.time()

        train_metrics = train(
            dset_train_split, model_outer_encoder, model_outer_decoder, model_inner_ae, criterion, optimizer, metrics, config, scheduler, voxelizer, model_inner_ae_ema, scaler,
            latent_mean, latent_std, latent_min, latent_max)

        val_metrics = val(
            dset_train_split, model_outer_encoder, model_outer_decoder, model_inner_ae_ema.module, criterion, metrics, config, voxelizer, epoch,
            latent_mean, latent_std, latent_min, latent_max)

        is_best = False
        if best_miou < val_metrics["miou_latent"]:
            is_best = True
            best_miou = val_metrics["miou_latent"]

        save_checkpoint({
            "epoch": epoch + 1,
            "config": config,
            "state_dict_ema": model_inner_ae_ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, is_best=is_best, config=config, epoch_num=epoch)

        print_metrics(epoch, train_metrics, val_metrics, time.time()-t0)

        if config["wandb"] > 0:
            wandb.log({"train": train_metrics, "val": val_metrics})


def train(dset_train_split, model_outer_encoder, model_outer_decoder, model_inner_ae, criterion, optimizer, metrics, config, scheduler, voxelizer, model_inner_ae_ema, scaler, latent_mean, latent_std, latent_min, latent_max):        
    metrics.reset()

    model_outer_encoder.eval()
    model_outer_decoder.eval()
    model_inner_ae.train()

    use_amp = config["amp"] > 0  

    # for each new epoch, randomly select a new data loader from the subset     
    subset_index = random.randrange(len(dset_train_split))	
    loader = torch.utils.data.DataLoader(
        dset_train_split[subset_index],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    for i, batch in enumerate(loader):     
        with torch.autocast(device_type=config["device"], dtype=torch.float16, enabled=use_amp):      
            voxels = voxelizer.forward(batch)
            z_latent, _, _ = model_outer_encoder(voxels)

            # normalize the input latent embeddings by the training set mean and std
            z_latent_shape = z_latent.shape
            z_latent = z_latent.view(z_latent.shape[0], -1)
            z_latent = (z_latent-latent_mean)/latent_std
            z_latent = z_latent.view(z_latent_shape)

            # add noise to the latent data 
            smooth_z_latent = z_latent + \
                torch.cuda.FloatTensor(z_latent.shape).normal_(0, config["smooth_sigma"])

            z_pred = model_inner_ae(smooth_z_latent)  

            del smooth_z_latent # clear variable to avoid running out of memory
            loss = criterion(z_pred, z_latent)

            # undo normalization before passing the denoised latent to the decoder
            z_latent_shape = z_latent.shape
            z_latent = z_latent.view(z_latent.shape[0], -1)
            z_pred = z_pred.view(z_latent.shape[0], -1)
    
            z_latent = z_latent * latent_std + latent_mean
            z_pred = z_pred * latent_std + latent_mean
        
            z_latent = z_latent.view(z_latent_shape)
            z_pred = z_pred.view(z_latent_shape)

            pred = model_outer_decoder(z_pred, None) 
            del z_pred # clear variable to avoid running out of memory
            pred_original = model_outer_decoder(z_latent, None)  
            del z_latent # clear variable to avoid running out of memory

        # backward
        if scaler is not None:  
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()   
        model_inner_ae_ema.update(model_inner_ae)

        if config["scheduler"] > 0:
            scheduler.step()

        metrics.update(loss, (pred, pred_original), voxels)
        
        del pred, pred_original, voxels # clear variable to avoid running out of memory

        if config["debug"] and i == 10:
            break

    return metrics.compute() 


def val(dset_val_split, model_outer_encoder, model_outer_decoder, model_inner_ae_ema, criterion, metrics, config, voxelizer, epoch, latent_mean, latent_std, latent_min, latent_max):
   
    metrics.reset()
    model_outer_encoder.eval()
    model_outer_decoder.eval()
    model_inner_ae_ema.eval()

    use_amp = config["amp"] > 0  

    # for each new epoch, randomly select a new data loader from the subset     
    subset_index = random.randrange(len(dset_val_split))	
    loader = torch.utils.data.DataLoader(
        dset_val_split[subset_index],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    
    with torch.autocast(device_type=config["device"], dtype=torch.float16, enabled=use_amp):
        with torch.no_grad():
            for i, batch in enumerate(loader):
                voxels = voxelizer(batch)
                z_latent, _, _ = model_outer_encoder(voxels)  
                
                # normalize the input latent embeddings by the training set mean and std
                z_latent_shape = z_latent.shape
                z_latent = z_latent.view(z_latent.shape[0], -1)
                z_latent = (z_latent-latent_mean)/latent_std
                z_latent = z_latent.view(z_latent_shape)

                # add noise to the latent data 
                smooth_z_latent = z_latent + \
                    torch.cuda.FloatTensor(z_latent.shape).normal_(0, config["smooth_sigma"])

                z_pred = model_inner_ae_ema(smooth_z_latent)
                
                del smooth_z_latent
                loss = criterion(z_pred, z_latent) # clear variable to avoid running out of memory

                #  undo normalization before passing the denoised latent to the decoder
                z_latent_shape = z_latent.shape
                z_latent = z_latent.view(z_latent.shape[0], -1)
                z_pred = z_pred.view(z_latent.shape[0], -1)
        
                z_latent = z_latent * latent_std + latent_mean
                z_pred = z_pred * latent_std + latent_mean
        
                z_latent = z_latent.view(z_latent_shape)
                z_pred = z_pred.view(z_latent_shape)

                pred = model_outer_decoder(z_pred, None) 
                del z_pred # clear variable to avoid running out of memory
                pred_original = model_outer_decoder(z_latent, None)  
                del z_latent # clear variable to avoid running out of memory

                # update metrics
                metrics.update(loss, (pred, pred_original), voxels)

                del pred, pred_original, voxels # clear variable to avoid running out of memory

                if config["debug"] and i == 10:
                        break
    
    return metrics.compute()


def print_metrics(epoch, train_metrics, val_metrics, time):
    """Print all metrics

    Args:
        epoch (int): epoch numbre
        train_metrics (list of Metrics): list train of metrics
        val_metrics (list of Metrics): list validation of metrics
        time (float): time (s)
    """
    str_ = f">> epoch: {epoch} ({time:.2f}s)"
    for (split, metric) in zip(["train", "valid"], [train_metrics, val_metrics]):
        if metric is None:
            continue
        str_ += "\n"
        str_ += f"[{split}]"
        for k, v in metric.items():
            if k == "loss":
                str_ += f" | {k}: {v:.8f}"
            else:
                str_ += f" | {k}: {v:.4f}"
    print(str_)


def compute_latent_stats(dset_train, model_outer_encoder, config, voxelizer):
    """
    Compute the mean and std of the latent embeddings 
    output by the frozen Compression model 
    to normalize the embeddings to 0 mean and std 1 
    """
    
    # if the latent statistics have already been computed, load them from the file
    if config["dset_name"] == "qm9":
        dataset_subset = [1]  # don't do subsampling
    else:
        dataset_subset = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  
    generator = torch.Generator().manual_seed(config["seed"])
    dset_train_split = torch.utils.data.random_split(dset_train, dataset_subset, generator=generator)

    loader_train = torch.utils.data.DataLoader(
        dset_train_split[0],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    
    mean_file_name = os.path.join(os.path.dirname(config["pretrained_path"]), "channel_latent_mean.npy")
    std_file_name = os.path.join(os.path.dirname(config["pretrained_path"]), "channel_latent_std.npy")
    min_file_name = os.path.join(os.path.dirname(config["pretrained_path"]), "channel_latent_min.npy")
    max_file_name = os.path.join(os.path.dirname(config["pretrained_path"]), "channel_latent_max.npy")

    if os.path.exists(mean_file_name) and os.path.exists(std_file_name) and os.path.exists(min_file_name) and os.path.exists(max_file_name):  
        latent_mean = np.load(mean_file_name)
        latent_std = np.load(std_file_name)
        latent_min = np.load(min_file_name)
        latent_max = np.load(max_file_name)
        print(">> loading latent statistics from " + mean_file_name)
        return latent_mean, latent_std, latent_min, latent_max
        
    else:
        print(">> computing latent statistics...")
        with torch.no_grad():
            for i, batch in enumerate(loader_train):
                voxels = voxelizer(batch)
                z_latent, _, _ = model_outer_encoder(voxels) 
                del voxels

                if i == 0:
                    z_latent_stack = (z_latent.view(config["batch_size"], -1)).detach().cpu()
                else:
                    z_latent_stack = torch.vstack((z_latent_stack, (z_latent.view(config["batch_size"], -1)).detach().cpu() ))

                del z_latent

                if i >= 10:
                    break

            latent_mean = z_latent_stack.mean(axis=0)
            latent_std = z_latent_stack.std(axis=0)
            latent_min = z_latent_stack.min(axis=0)[0]  # take values, ignore indices
            latent_max = z_latent_stack.max(axis=0)[0]

            # save to file
            if torch.is_tensor(latent_mean):
                latent_mean = latent_mean.cpu().numpy()
                latent_std = latent_std.cpu().numpy()
                latent_min = latent_min.cpu().numpy()
                latent_max = latent_max.cpu().numpy()
            
            np.save(mean_file_name, latent_mean)
            np.save(std_file_name, latent_std)
            np.save(min_file_name, latent_min)
            np.save(max_file_name, latent_max)

            return latent_mean, latent_std, latent_min, latent_max

if __name__ == "__main__":
    # parse args
    config = parse_args()

    # basic inits
    torch.set_default_dtype(torch.float32)
    print(">> training on device:", config["device"])
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if config["device"] == "cuda":
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True

    # create dir to save outputs of experiment
    create_exp_dir(config)
    create_exp_name(config)

    # start wandb
    if config["wandb"] > 0:
        wandb.init(
            project="voxmol",
            config=config,
            name=config["exp_name"],
            dir=config["output_dir"]
        )

    # start training
    main(config)
