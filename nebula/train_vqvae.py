import time
import random
import wandb
import torch
import torchmetrics
import torch.backends.cudnn as cudnn

from options import parse_args

from dataset.dataset import DatasetVoxMol
from voxelizer import Voxelizer

from models import create_model
from models.ema import ModelEma  
from models.adamw import AdamW   

from metrics import MetricsDenoise
from utils.utils import load_checkpoint, save_checkpoint, create_exp_dir, create_exp_name


def main(config):
    '''
    This script trains the compression model used to obtain the latent embeddings for sampling new molecules.

    python train_vqvae.py \
    --model_config models/configs/vqvae_config.yml \
    --exp_name vqvae/ 

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
    model = create_model(config)
    if torch.cuda.device_count() > 1:  
        model = torch.nn.DataParallel(model)

    model.to(config["device"])

    if config["pretrained_path"]:
        model, _ = load_checkpoint(model, config["pretrained_path"])
    else:
        print(">> Model is running with random initialization...")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">> model has {(n_params/1e6):.02f}M parameters")

    optimizer = AdamW(  
        model.parameters(),
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

    model_ema = ModelEma(model, decay=.998)

    metrics = MetricsDenoise(
        loss=torchmetrics.MeanMetric(),
        miou=torchmetrics.classification.BinaryJaccardIndex(),
    )
    metrics.to(config["device"])
    best_miou = 0

    print(">> start training...")

    for epoch in range(0, config["num_epochs"]):
        t0 = time.time()

        # train
        train_metrics = train(
            dset_train_split, model, optimizer, metrics, config, scheduler, voxelizer, model_ema, scaler
        )

        # val
        val_metrics = val(
            dset_val_split, model_ema.module, metrics, config, voxelizer, epoch
        )
        
        # save model, print metrics, log wandb
        is_best = False
        if best_miou < val_metrics["miou"]:
            is_best = True
            best_miou = val_metrics["miou"]

        save_checkpoint({
            "epoch": epoch + 1,
            "config": config,
            "state_dict_ema": model_ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, is_best=is_best, config=config) 

        print_metrics(epoch, train_metrics, val_metrics, time.time()-t0)

        if config["wandb"] > 0:
            wandb.log({"train": train_metrics, "val": val_metrics})


def train(dset_train_split, model, optimizer, metrics, config, scheduler, voxelizer, model_ema, scaler):
    """Train one epoch of model on loader"""
    
    metrics.reset()
    model.train()

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
            
            # forward and backward
            vq_loss, pred, perplexity = model(voxels)
            rec_loss = torch.nn.functional.mse_loss(pred, voxels)
            loss = rec_loss + vq_loss

        if scaler is not None:  
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        model_ema.update(model)

        if config["scheduler"] > 0:
            scheduler.step()
    
        metrics.update((loss, rec_loss, vq_loss), pred, voxels)

        if config["debug"] and i == 10:
            break

    return metrics.compute() 


def val(dset_val_split, model, metrics, config, voxelizer, epoch):
    """Evaluate model on loader"""
    metrics.reset()
    model.eval()

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
                # voxelize
                voxels = voxelizer(batch)

                # forward
                vq_loss, pred, perplexity = model(voxels)
                rec_loss = torch.nn.functional.mse_loss(pred, voxels)
                loss = rec_loss + vq_loss

                metrics.update((loss, rec_loss, vq_loss), pred, voxels)

                if config["debug"] and i == 1:
                    break
            
    return metrics.compute() 


def print_metrics(epoch, train_metrics, val_metrics, time):
    """Print all metrics

    Args:
        epoch (int): epoch number
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


if __name__ == "__main__":
    config = parse_args()

    torch.set_default_dtype(torch.float32)
    print(">> training on device:", config["device"])
    torch.set_default_dtype(torch.float32)  # NOTE
    seed = config["seed"]
    torch.manual_seed(seed)
    # np.random.seed(seed)
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

    main(config)
