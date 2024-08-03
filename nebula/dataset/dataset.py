import os
import math
import random
import torch
from torch.utils.data import Dataset

from utils.utils import elements_hash

class DatasetVoxMol(Dataset):
    """
    A custom dataset class for VoxMol dataset.

    Args:
        dset_name (str): The name of the dataset. Default is "drugs".
        data_dir (str): The directory path where the dataset is located. Default is "dataset/data/".
        elements (list): The list of elements to include in the dataset. Default is None.
        split (str): The split of the dataset to use. Must be one of "train", "val", or "test". Default is "train".
        rotate (bool): Whether to apply rotation augmentation to the coordinates. Default is True.
        shift (int): The amount of shift augmentation to apply to the coordinates. Default is 0.
        small (bool): Whether to use a small subset of the dataset. Default is False.
        atomic_radius (float): The atomic radius to assign to each atom. Default is 0.5.
        max_n_atoms (int): The maximum number of atoms allowed in a sample. Default is 80.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index: int): Returns the sample at the given index.

    """

    def __init__(
        self,
        dset_name: str = "drugs",
        data_dir: str = "dataset/data/",
        elements: list = None,
        split: str = "train",
        rotate: bool = True,
        shift: int = 0.5,  
        small: bool = False,
        atomic_radius: float = .5,  
        max_n_atoms: int = 80,
        multi_file=False,
    ):
        if elements is None:
            elements = elements_hash
        assert split in ["train", "val", "test"], "split must be train, val or test"
        self.dset_name = dset_name
        self.data_dir = data_dir
        self.split = split
        self.rotate = rotate
        self.shift = shift  
        self.atomic_radius = atomic_radius  
        self.max_n_atoms = max_n_atoms

        
        if multi_file:  # train on multiple smaller files
            data_full = torch.load(os.path.join(data_dir))
        else:
            data_full = torch.load(os.path.join(data_dir, dset_name, f"{split}_data.pth"))
        self.data = data_full
        if small:
            self.data = self.data[:5000]

        # Add any extra data preprocessing if needed
        if max_n_atoms > 0:
            self._filter_by_n_atoms()
        self._filter_by_elements(elements)

    def _filter_by_elements(self, elements: list):
        """
        Filter the dataset by a list of elements.

        Args:
            elements (list): List of elements to filter by.

        Returns:
            None
        """
        filtered_data = []
        
        elements_ids = [elements_hash[element] for element in elements]

        for datum in self.data:

            atoms = datum['atoms_channel'][datum['atoms_channel'] != 999]
            include = True

            for atom_id in atoms.unique():
                if int(atom_id.item()) not in elements_ids:
                    include = False
                    break
            if include:
                filtered_data.append(datum)
        
        if len(self.data) != len(filtered_data):
            print(f"  | filter data (elements): data reduced from {len(self.data)} to {len(filtered_data)}")
            self.data = filtered_data

    def _filter_by_n_atoms(self):
        """
        Filters the dataset by the number of atoms in each datum.

        Returns:
            None
        """
        filtered_data = []
        for datum in self.data:
            n_atoms = len(datum["atoms_channel"][datum["atoms_channel"] != 999])
            if n_atoms < self.max_n_atoms:
                filtered_data.append(datum)
        if len(self.data) != len(self.data):
            print(f"  | filter data (n atoms): data reduced from {len(self.data)} to {len(filtered_data)}")
            self.data = filtered_data

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Retrieves the sample at the given index from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the sample data, including the coordinates, atoms channel, and radius.
        """
        sample_raw = self.data[index]

        sample = {
            "coords": sample_raw["coords"],
            "atoms_channel": sample_raw["atoms_channel"],
            "radius": torch.Tensor(sample_raw["atoms_channel"].shape).fill_(self.atomic_radius),
            "smiles": sample_raw["smiles"],
        }

        # Add noise/rotation on the coords, or any other augmentation
        sample = _center_coords(sample)
        if self.rotate:
            sample = _rotate_coords(sample)
        if self.shift > 0:
            sample = _shift_coords(sample, delta=self.shift)

        return sample


########################################################################################
# Data augmentation
def _center_coords(sample: dict):  
    """
    Center the coordinates of the atoms in the sample.

    Args:
        sample (dict): A dictionary containing the sample data.

    Returns:
        dict: The modified sample with centered coordinates.
    """
    coords = sample["coords"]
    
    if len(sample["coords"].shape) == 3:
        coords = sample["coords"]
        coords = torch.squeeze(coords)  # remove first dimension
        sample["coords"] = [] 
        sample["coords"] = coords 
            
    mask = sample['atoms_channel'] != 999
    coords_masked = coords[mask]  # ignore value 999

    # go to center of mass
    center_coords = torch.mean(coords_masked, dim=0)
    center_coords = center_coords.unsqueeze(0).repeat(coords_masked.shape[0], 1)
    coords_masked = coords_masked - center_coords

    sample["coords"][mask] = coords_masked
    return sample


def _shift_coords(sample: dict, delta: float = 1.):  
    """
    Shifts the coordinates of atoms in the sample by adding random noise.

    Args:
        sample (dict): The input sample containing the atoms and coordinates.
        delta (float, optional): The maximum magnitude of the random noise. Defaults to 1.

    Returns:
        dict: The modified sample with shifted coordinates.
    """
    mask = sample['atoms_channel'] != 999
    noise = (torch.rand((1, 3)) - 1/2)*2*delta
    sample["coords"][mask] += noise.repeat(sample["coords"][mask].shape[0], 1)
    return sample


def _rotate_coords(sample: dict):
    """
    Rotate the coordinates of a sample using a random rotation matrix.

    Args:
        sample (dict): A dictionary containing the sample data.

    Returns:
        dict: The modified sample dictionary with rotated coordinates.
    """
    rot_matrix = _random_rot_matrix()

    coords = sample["coords"]

    idx = sample['atoms_channel'] != 999
    coords_masked = coords[idx]  # ignore value 999
    coords_masked = torch.reshape(coords_masked, (-1, 3))

    # go to center of mass
    center_coords = torch.mean(coords_masked, dim=0)
    center_coords = center_coords.unsqueeze(0).tile((coords_masked.shape[0], 1))
    coords_masked = coords_masked - center_coords

    coords_rot = torch.einsum("ij, kj -> ki", rot_matrix, coords_masked)
    coords[: coords_rot.shape[0], :] = coords_rot
    sample["coords"] = coords
    return sample


def _random_rot_matrix():
    """
    Generate a random rotation matrix.

    Returns:
        torch.Tensor: Random rotation matrix.
    """
    theta_x = random.uniform(0, 2) * math.pi  
    rot_x = torch.Tensor([
        [1, 0, 0],
        [0, math.cos(theta_x), -math.sin(theta_x)],
        [0, math.sin(theta_x), math.cos(theta_x)],
    ])
    theta_y = random.uniform(0, 2) * math.pi
    rot_y = torch.Tensor([
        [math.cos(theta_y), 0, -math.sin(theta_y)],
        [0, 1, 0],
        [math.sin(theta_y), 0, math.cos(theta_y)],
    ])
    theta_z = random.uniform(0, 2) * math.pi
    rot_z = torch.Tensor([
        [math.cos(theta_z), -math.sin(theta_z), 0],
        [math.sin(theta_z), math.cos(theta_z), 0],
        [0, 0, 1],
    ])

    return rot_z @ rot_y @ rot_x


########################################################################################
# Test
if __name__ == "__main__":
    from voxmol.voxelizer import Voxelizer
    from voxmol.utils import makedir, visualize_voxel_grid
    ROTATE = True
    TRANS = 1.
    dset = DatasetVoxMol(dset_name="drugs", data_dir="data/", rotate=ROTATE, shift=TRANS)
    voxelizer = Voxelizer(grid_dim=32)
    for i in range(len(dset)):
        sample = dset[i]
        sample["coords"] = sample["coords"].unsqueeze(0)
        sample["atoms_channel"] = sample["atoms_channel"].unsqueeze(0)
        sample["radius"] = sample["radius"].unsqueeze(0)

        voxel = voxelizer(sample)
        dirname = f"figs/rot{ROTATE}_trans{TRANS}/"
        makedir(dirname)
        visualize_voxel_grid(voxel, fname=f"{dirname}/{i}.png", to_png=True, to_html=False)
        if i == 2:
            break

