import os
import numpy as np
import glob
from rdkit import Chem
import torch

from nebula.utils.rdkit_functions import Molecule, Molecule2D


def process_obabel(output_dir="output"):
    """
    convert xyz to sdf
    
    """               
    curr_dir = os.getcwd()
    path_xyzs1 = os.path.join(curr_dir, output_dir, "xyzs")
    path_xyzs2 = os.path.join(curr_dir, output_dir)
    cmd = f"cd {path_xyzs1}; obabel *xyz -osdf -O {path_xyzs2}/molecules_obabel.sdf --title  end"
    os.system(cmd)
    os.chdir(curr_dir)


atom_encoder_dict = {'qm9_with_h': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
                     'qm9_no_h': {'C': 0, 'N': 1, 'O': 2, 'F': 3},
                     'geom_with_h': {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,'P': 8,
                                     'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15},
                     'geom_no_h': {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Al': 5, 'Si': 6, 'P': 7, 'S': 8,
                                   'Cl': 9, 'As': 10,'Br': 11, 'I': 12, 'Hg': 13, 'Bi': 14}}

atom_decoder_dict = {'qm9_with_h': ['H', 'C', 'N', 'O', 'F'],
                     'qm9_no_h': ['C', 'N', 'O', 'F'],
                     'geom_with_h': ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br',
                                     'I', 'Hg', 'Bi'],
                     'geom_no_h': ['B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I',
                                   'Hg', 'Bi']}

def open_babel_preprocess(file, name, save_imgs=0):
    """
    :param file: str
    :param name: 'qm9_with_h', 'qm9_no_h, 'geom_with_h', 'geom_no_h'
    :return:
    """

    filt_max_mol = 1
    sanitize_mol = 1

    smiles_generated_save = []

    if save_imgs:
        from rdkit.Chem import Draw
        dirname_smiles_png = file.replace("molecules_obabel.sdf", "generated_smiles_png/")
        if os.path.exists(dirname_smiles_png) == False:
            os.mkdir(dirname_smiles_png)  

    smile_counter = -1
    
    atom_encoder = atom_encoder_dict[name]
    atom_decoder = atom_decoder_dict[name]

    with open(file, "r") as f:
        lines = f.readlines()[3:]

    result = []
    temp = []

    for line in lines:
        line = line.strip()

        if not line or "M" in line or "$" in line or "OpenBabel" in line:
            continue

        vec = line.split()
        if vec != ['end']:
            temp.append(vec)
        else:
            result.append(temp)
            temp = []

    all_mols = []
    all_mol_idx = [] 

    for k, array in enumerate(result):
        smile_counter += 1

        if k == 10_000:
            break
        atom_temp = []
        pos_temp = []
        new_pos = []
        col = row = array[0][0]
        skip = False
        for i in range(int(col)):
            try:
                element = array[i + 1][3]
            except:
                skip = True
                break
            
            x = atom_encoder.get(element, None)
            if x is None:
                pass
            atom_temp.append(x)
            x_pos = array[i + 1][0]
            x_pos = float(x_pos)
            y_pos = array[i + 1][1]
            y_pos = float(y_pos)
            z_pos = array[i + 1][2]
            z_pos = float(z_pos)
            pos_temp.append([x_pos, y_pos, z_pos])
        if skip:
            all_mol_idx.append(0)
            continue
            
        new_pos.append(pos_temp)

        skip = False
        iteration = array[0][1]
        cols, rows = int(col), int(row)
        matrix = [[0 for x in range(cols)] for y in range(rows)]
        for j in range(int(iteration)):
            d = j + int(col) + 1
            a = int(array[d][0]) - 1
            b = int(array[d][1]) - 1
            c = int(array[d][2])
            try:
                matrix[a][b] = c
                matrix[b][a] = c
            except:
                skip = True
                break
        if skip:
            all_mol_idx.append(0)
            continue
        
        X = torch.tensor(atom_temp)
        charges = torch.zeros(X.shape)
        E = torch.tensor(matrix)
        posis = torch.tensor(new_pos[0])
        molecule = Molecule(atom_types=X, bond_types=E, positions=posis, charges=charges, atom_decoder=atom_decoder)
        
        if filt_max_mol:
            rdmol = molecule.rdkit_mol
            mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
            if len(mol_frags) > 1:
                largest_mol = max(mol_frags, default=molecule, key=lambda m: m.GetNumAtoms())
            else:
                largest_mol = rdmol

            if sanitize_mol:
                try:
                    largest_mol_not_sanitized = largest_mol
                    largest_mol = Chem.MolFromSmiles(Chem.MolToSmiles(largest_mol_not_sanitized), sanitize=True) 
                    largest_mol= Chem.AddHs(largest_mol)
                except:
                    largest_mol = largest_mol_not_sanitized
                atom_temp_max = []
                for atom in largest_mol.GetAtoms():
                    element = atom.GetSymbol()
                    x = atom_encoder.get(element, None)
                    if x is None:
                        pass
                    atom_temp_max.append(x)
                
                bond_types_list = ['ZERO', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
                matrix = np.zeros((largest_mol.GetNumAtoms(), largest_mol.GetNumAtoms()))
                for bond in largest_mol.GetBonds():
                    b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    bond_type = bond_types_list.index(bond.GetBondType().name)
                    matrix[b][e] = bond_type
                    matrix[e][b] = bond_type
                X_max = torch.tensor(atom_temp_max)
                charges_max = torch.zeros(X_max.shape)
                E_max = torch.LongTensor(matrix)
                
                molecule_max = Molecule2D(atom_types=X_max, bond_types=E_max, charges=charges_max, atom_decoder=atom_decoder)
                molecule_max.build_molecule(atom_decoder=atom_decoder)
                
            else:
                atom_temp_max = []
                pos = largest_mol.GetConformer(largest_mol.GetConformers()[0].GetId()).GetPositions()
                for atom in largest_mol.GetAtoms():
                    element = atom.GetSymbol()
                    x = atom_encoder.get(element, None)
                    if x is None:
                        pass
                    atom_temp_max.append(x)
                    
                bond_types_list = ['ZERO', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
                matrix = np.zeros((largest_mol.GetNumAtoms(), largest_mol.GetNumAtoms()))
                for bond in largest_mol.GetBonds():
                    b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    bond_type = bond_types_list.index(bond.GetBondType().name)
                    matrix[b][e] = bond_type
                    matrix[e][b] = bond_type
                
                X_max = torch.tensor(atom_temp_max)
                charges_max = torch.zeros(X_max.shape)
                E_max = torch.LongTensor(matrix)
                posis_max = torch.tensor(pos)
                
                molecule_max = Molecule(atom_types=X_max, bond_types=E_max, positions=posis_max, charges=charges_max, atom_decoder=atom_decoder)

                molecule_max.build_molecule(atom_decoder=atom_decoder)
        
        else:
            molecule_max = molecule

        all_mols.append(molecule_max)
        all_mol_idx.append(1)
        
        try:
            molecule_max_rdkit = molecule_max.rdkit_mol
            molecule_max_rdkit = Chem.MolFromSmiles(Chem.MolToSmiles(molecule_max_rdkit))
            smiles_generated = Chem.MolToSmiles(molecule_max_rdkit) 

            if save_imgs:
                smiles_img = Draw.MolToImage(molecule_max_rdkit)
                smiles_img.save(dirname_smiles_png + f"sample_{smile_counter:05d}.png")

            smiles_generated_save.append(smiles_generated)
        except:
            pass

    return smiles_generated_save