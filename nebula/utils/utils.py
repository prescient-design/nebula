from datetime import datetime
import numpy as np
import os
import shutil
import torch
import yaml

from pyuul.sources.globalVariables import *
from pyuul.sources import  hashings
from pyuul.sources.globalVariables import PADDING_INDEX

elements_hash = {
    "C": 0,
    "H": 1,
    "O": 2,
    "N": 3,
    "F": 4,
    "S": 5,
    "Cl": 6,
    "Br": 7,
    "P": 8,
    "I": 9,
    "B": 10,
}

radiusSingleAtom = {
    "MOL": {
        "C": 2.0,
        "H": 1.0,
        "O": 1.6,
        "N": 1.75,
        "F": 1.545,
        "S": 2.0,
        "Cl": 2.045,
        "Br": 2.165,
        "P": 2.1,
        "I": 2.36,
        "B": 2.04,
    }
}

def makedir(path):
    """Create a directory

    Args:
        path (str): path to dir
    """
    if not os.path.exists(path):
        os.makedirs(path)


def create_exp_name(config):
    """Create the experiment name

    Args:
        config (dcit): dict containing the arguments for the experiment
    """
    config["exp_name"] = f"exp_{config['dset_name']}_" + f"lr{config['lr']}" 
        
    
def create_exp_dir(config):
    """Create the directory for the experiment/run

    Args:
        config (dcit): dict containing the arguments for the experiment
    """
    if config['exp_name'] is None:
        exp_name = datetime.now().strftime('%Y%m%d_%H%M%S').replace("'", '')
    else:
        exp_name = config['exp_name']
    output_dir = os.path.join(config['exp_dir'], exp_name)
    config['output_dir'] = output_dir
    makedir(output_dir)
    print('>> saving experiments in:', output_dir)
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


def save_checkpoint(state, config, is_best=False, epoch_num=0):
    """Save a model checkpoint

    Args:
        state (list of torch.Tensor): state of a neural network
        is_best (bool): save best model if True
        config (dict): dict containing arguments of the experiment
    """
    filename = os.path.join(config['output_dir'], 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, os.path.join(config['output_dir'], 'best_checkpoint.pth.tar')
        )


def load_checkpoint(model, pretrained_path):
    """Load a model checkopint

    Args:
        model (torch.nn.model): model with random weights
        pretrained_path (str): path to pretrained experiment

    Returns:
        model: the model with pretrained weights loaded
    """
    checkpoint = torch.load(pretrained_path)
    print("loading checkpoint from epoch " + str(checkpoint["epoch"]))
    sd = "state_dict_ema" if "state_dict_ema" in checkpoint else "state_dict"

    # to cope with torch.compile and multi-GPU training
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint[sd].items()}
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint[sd].items()}

    model_dict = model.state_dict()

    # update the model weights without the new mlp layers
    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}        
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict) 

    return model, int(str(checkpoint["epoch"]))


def mol2xyz(sample):
    sample = remove_atoms_too_close(sample)
    n_atoms = sample['atoms_channel'].shape[-1]
    atom_elements = ["C", "H", "O", "N", "F", "S", "Cl", "Br", "P", "I", "B"]
    xyz_str = str(n_atoms) + "\n\n"
    for i in range(n_atoms):
        element = sample['atoms_channel'][0, i]
        element = atom_elements[int(element.item())]

        coords = sample['coords'][0, i, :]

        line = element + "\t" + str(coords[0].item()) + "\t" + str(coords[1].item()) + "\t" + str(coords[2].item())
        xyz_str += line + "\n"
    return xyz_str


def atomlistToRadius(atomList, hashing, device="cpu"):
    """Convert atomlist to a list of radius specified by hashing.
     Modified from original PyUUL code
    Args:
        atomList (list): list of atom types
        hashing (dict): dict from atom type to radius (in A)
        device (str, optional): cpu or gpu. Defaults to "cpu".

    Returns:
        _type_: list of set of radius. One for each molecule.
    """

    radius = []
    for singleAtomList in atomList:
        haTMP = []
        for i in singleAtomList:
            resname, atName = i.split("_")[0], i.split("_")[2]
            if resname in hashing and atName in hashing[resname]:
                haTMP += [hashing[resname][atName]]
            else:
                haTMP += [1.0]
        radius += [torch.tensor(haTMP, dtype=torch.float, device=device)]
    radius = torch.torch.nn.utils.rnn.pad_sequence(radius, batch_first=True, padding_value=PADDING_INDEX)
    return radius


def remove_atoms_too_close(mol, dist_thr=.8):
    idxs = [None]
    while len(idxs) > 0:
        dists = torch.cdist(mol["coords"], mol["coords"], compute_mode="donot_use_mm_for_euclid_dist")[0]
        idxs = np.where((dists > 0) & (dists < dist_thr))
        idxs = list(set(np.concatenate(idxs)))
        if len(idxs) == 0:
            break
        n_atoms = mol["atoms_channel"].shape[-1]
        rows = torch.BoolTensor(n_atoms).fill_(True)
        rows[idxs[0]] = False
        mol = {
            "coords": mol["coords"][:, rows, :],
            "atoms_channel": mol["atoms_channel"][:, rows],
            "radius": mol["radius"][:, rows],
        }

    return mol


def parseSDF(SDFFile):
    """
        function to parse pdb files. It can be used to parse a single file or all the pdb files in a folder. In case a folder is given, the coordinates are gonna be padded

        Parameters
        ----------
        SDFFile : str
        path of the PDB file or of the folder containing multiple PDB files

        Returns
        -------
        coords : torch.Tensor
        coordinates of the atoms in the pdb file(s). Shape ( batch, numberOfAtoms, 3)

        atomNames : list
        a list of the atom identifier. It encodes atom type, residue type, residue position and chain

    """
    if not os.path.isdir(SDFFile):
        fil = SDFFile
        totcoords=[]
        totaname=[]
        coords = []
        atomNames = []
        for line in open(fil).readlines():
            a=line.strip().split()
            if len(a)==16: ## atom
                element = a[3]
                try:
                    x = float(a[0])
                    y = float(a[1])
                    z = float(a[2])
                    coords += [[x,y,z]]
                    aname = "MOL"+"_"+"0"+"_"+element+"_"+"A"

                    atomNames += [aname]
                except:
                    print("error with file: " + SDFFile)
            elif "$$$$" in line:
                totcoords+=[torch.tensor(coords)]
                totaname += [atomNames]
                coords=[]
                atomNames=[]
        return torch.torch.nn.utils.rnn.pad_sequence(totcoords, batch_first=True, padding_value=PADDING_INDEX),totaname
    else:
        totcoords = []
        totaname = []
        for fil in sorted(os.listdir(SDFFile)):
            coords = []
            atomNames = []
            for line in open(SDFFile+fil).readlines():
                a = line.strip().split()
                if len(a) == 16:  ## atom
                    element = a[3]
                    x = float(a[0])
                    y = float(a[1])
                    z = float(a[2])
                    coords += [[x, y, z]]
                    aname = "MOL"+"_"+"0"+"_"+element+"_"+"A"

                    atomNames += [aname]
                elif "$$$$" in line:
                    totcoords += [torch.tensor(coords)]
                    totaname += [atomNames]
                    coords = []
                    atomNames = []
        return torch.torch.nn.utils.rnn.pad_sequence(totcoords, batch_first=True, padding_value=PADDING_INDEX),totaname
