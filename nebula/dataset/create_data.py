# inspired by:
#   - https://github.com/DeepGraphLearning/ConfGF and
#   - https://github.com/cvignac/MiDi/blob/master/src/datasets/qm9_dataset.py


import argparse
import gc
import os
import torch
from tqdm import tqdm

from pyuul import utils
from rdkit import Chem
from rdkit import RDLogger

from rdkit.Chem import (
    AllChem, Descriptors, Descriptors3D, rdFMCS, rdMolAlign, rdMolDescriptors)
from rdkit.Geometry import rdGeometry

from utils.utils import atomlistToRadius
from utils.utils import elements_hash, radiusSingleAtom

from pyuul.sources.globalVariables import *

RDLogger.DisableLog("rdApp.*")

"""
Pre-process the dataset to run generation on new input molecules. Create a pth file, given an sdf file 

Optionally convert input SMILES to 3D conformers 

Args:
    data_dir (str): Directory with input sdf file (or where sdf file is generated from input smiles). 
    input_smiles (str): Optional input SMILES sequence to be processed as a seed lead compound
    sdf_file (str): Default path to save the sdf file with 3D conformers generated for an input SMILES sequence
    
"""

def preprocess_dataset(data_dir):   
    # elements hash, radii and bond types
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
    # using tha atomic radii from https://github.com/gnina/libmolgrid/blob/master/src/atom_typer.cpp
    # which is the same used in AutoDock v4.
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

    num_errors = 0
    len_data = 0
    for sdf_file in os.listdir(os.path.join(data_dir)):     
        # save each sdf file separately 
        sdf_file_path_save = os.path.join(data_dir, sdf_file.replace(".sdf", ".pth"))
        if os.path.exists(sdf_file_path_save) == False:
            count_errors = 0
            data = [] 
            sdf_file_path = os.path.join(data_dir, sdf_file)
            suppl = Chem.SDMolSupplier(sdf_file_path, removeHs=False, sanitize=False)
            coords, atname = utils.parseSDF(sdf_file_path)
            atoms_channel = utils.atomlistToChannels(atname, hashing=elements_hash)
            radius = atomlistToRadius(atname, hashing=radiusSingleAtom)
            
            for i, mol in enumerate(tqdm(suppl)):
                try:
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
                    if smiles is None:
                        num_errors += 1

                    datum = {
                        "mol": mol,
                        "smiles": smiles,
                        "coords": coords[i].clone(),
                        "atoms_channel": atoms_channel[i].clone(),
                        "radius": radius[i].clone(),
                        # "target": mol.GetPropsAsDict()['Y']  # add if the sdf has a label / target
                    }
                    
                    data.append(datum)
                except:
                    count_errors += 1
                    print("Error occured " + str(count_errors) + " times")
            
            # save each sdf file separately             
            torch.save(data, sdf_file_path_save,)
            len_data += len(data)
    print(f"  >> split size: {len_data} ({num_errors} errors)")
    return 


def smiles2sdf(input_smiles, sdf_file):
    """
    create conformers given an input smiles
    """   
    mol = Chem.MolFromSmiles(input_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(
        mol, 
        numConfs=50,
        pruneRmsThresh=0.01, 
        useSmallRingTorsions=True,
        useMacrocycleTorsions=True)
    
    # optimize conformers with force field
    opts = AllChem.MMFFOptimizeMoleculeConfs(
        mol, 
        nonBondedThresh=10, 
        ignoreInterfragInteractions=False,
        maxIters=200) 

    mol = Chem.RemoveHs(mol)

    # align conformers
    rdMolAlign.AlignMolConformers(mol)

    sdwriter = Chem.rdmolfiles.SDWriter(sdf_file)

    if len(mol.GetConformers()) > 0:
        for conf in mol.GetConformers():
            sdwriter.write(mol, conf.GetId())
    else:
        sdwriter.write(mol)
    sdwriter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/seed_dir/")
    args = parser.parse_args()

    data = {}, {}

    preprocess_dataset(args.data_dir)

    gc.collect()
