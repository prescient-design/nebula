import numpy as np
from copy import deepcopy
from functools import partial
from scipy import ndimage as ndi
import torch

from pyuul import VolumeMaker
from torch import nn

from utils.utils import mol2xyz


class Voxelizer(nn.Module):
    """
    Voxelizer module for converting molecular data into voxel representations.

    Args:
        grid_dim (int): The dimension of the voxel grid (default: 32).
        resolution (float): The resolution of the voxel grid (default: 0.25).
        cubes_around (int): The number of cubes around each atom in the voxel grid (default: 5).
        num_channels (int): The number of channels in the voxel grid (default: 5).
        device (str): The device to use for computation (default: "cpu").
    """

    def __init__(
            self,
            grid_dim: int = 32,
            resolution: float = 0.25,
            cubes_around: int = 5,
            num_channels: int = 5,  
            device: str = "cpu"
    ):
        super(Voxelizer, self).__init__()
        self.grid_dim = grid_dim
        self.device = device
        self.resolution = resolution
        self.cubes_around = cubes_around
        self.num_channels = num_channels

        self.vol_maker = VolumeMaker.Voxels(
            device=device,
            sparse=False,
        )

    def forward(self, batch: dict):
        return self.mol2vox(batch)

    def mol2vox(self, batch: dict):
        """
        Convert molecular data to voxel representation.

        Args:
            batch (dict): A dictionary containing the molecular data.

        Returns:
            torch.Tensor: The voxel representation of the molecular data.
        """
        # dumb coordinates to center molecule
        batch = self._add_dumb_coords(batch)  
        voxels = self.vol_maker(
                batch["coords"].to(self.device),
                batch["radius"].to(self.device),
                batch["atoms_channel"].to(self.device),
                resolution=self.resolution,
                cubes_around_atoms_dim=self.cubes_around,
                function="gaussian",
                numberchannels=self.num_channels,
                steepness=100
            )
        # get center box (remove dumb coordinates)
        c = voxels.shape[-1] // 2
        box_min, box_max = c - self.grid_dim // 2, c + self.grid_dim // 2
        voxels = voxels[:, :, box_min:box_max, box_min:box_max, box_min:box_max]
        return voxels

    
    def vox2mol(
        self,
        voxels: torch.Tensor,
        radius: float = 0.5,
        resolution: float = 0.25,
        refine: bool = True
    ):
        """
        Convert voxel representation back to molecular data.

        Args:
            voxels (torch.Tensor): The voxel representation.
            radius (float): The radius of the atoms in the molecular data (default: 0.5).
            resolution (float): The resolution of the voxel grid (default: 0.25).
            refine (bool): Whether to refine the coordinates (default: True).

        Returns:
            list: A list of dictionaries containing the refined molecular data.
        """
        assert len(voxels.shape) == 5
        # intialize coods with simple peak detection
        mol_inits = []
        voxel_inits = []
        for voxel in voxels:
            peaks = find_peaks(voxel.cpu())
            mol_init = get_atom_coords(peaks, rad=radius, resolution=resolution)
            if mol_init is not None and mol_init["coords"].shape[1] < 200:
                mol_inits.append(mol_init)
                voxel_inits.append(voxel.unsqueeze(0))

        if len(mol_inits) == 0:
            return None

        if not refine:
            return mol_inits
        
        voxel_inits = torch.cat(voxel_inits, axis=0)

        # refine coords
        optim_factory = partial(
            torch.optim.LBFGS, history_size=10, max_iter=4, line_search_fn="strong_wolfe",
        )   

        mols = self._refine_coords(mol_inits, voxel_inits, optim_factory, maxiter=10)
        del voxels, mol_inits, voxel_inits
        torch.cuda.empty_cache()

        return mols

    def _refine_coords(
        self,
        mol_inits: list,
        voxels: torch.Tensor,
        optim_factory: callable,
        tol: float = 1e-6,
        maxiter: int = 10
    ):
        """
        Refine the coordinates of the molecular data.

        Args:
            mol_inits (list): A list of dictionaries containing the initial molecular data.
            voxels (torch.Tensor): The voxel representation.
            optim_factory (function): A function that creates an optimizer.
            tol (float): The tolerance for convergence (default: 1e-6).
            maxiter (int): The maximum number of iterations (default: 10).
s
        Returns:
            list: A list of dictionaries containing the refined molecular data.
        """

        assert len(voxels.shape) == 5, "voxels need to have dimension 5 (including the batch dim.)"

        mols = []
        for i in range(voxels.shape[0]):
            mol_init = mol_inits[i]
            voxel = voxels[i].unsqueeze(0)

            mol = deepcopy(mol_init)
            mol["coords"].requires_grad = True

            optimizer = optim_factory([mol["coords"]])

            def closure():
                optimizer.zero_grad()
                voxel_fit = self.forward(mol)
                loss = torch.nn.functional.mse_loss(voxel, voxel_fit)  
                loss.backward()
                return loss

            loss = 1e10  
            for _ in range(maxiter):
                try:
                    prev_loss = loss  
                    loss = optimizer.step(closure)
                except Exception:
                    mol = deepcopy(mol_init)
                    break

                if abs(loss.item() - prev_loss) < tol:  
                    break

            mols.append({
                "coords": mol["coords"].detach().cpu(),
                "atoms_channel": mol["atoms_channel"].detach().cpu(),
                "radius": mol["radius"].detach().cpu(),
            })

        return mols

    def _add_dumb_coords(self, batch: dict):
        """
        Add dumb coordinates to center the molecule.

        Args:
            batch (dict): A dictionary containing the molecular data.

        Returns:
            dict: A dictionary containing the molecular data with dumb coordinates added.
        """
        bsz = batch['coords'].shape[0]
        return {
            "coords": torch.cat(
                (batch['coords'], torch.Tensor(bsz, 1, 3).fill_(-20), torch.Tensor(bsz, 1, 3).fill_(20)), 1
            ),
            "atoms_channel": torch.cat(
                (batch['atoms_channel'], torch.Tensor(bsz, 2).fill_(0)), 1
            ),
            "radius": torch.cat(
                (batch['radius'], torch.Tensor(bsz, 2).fill_(.5), ), 1
            )
        }


########################################################################################
# aux functions
def local_maxima(data: np.ndarray, order: int = 1):
    """
    Find local maxima in a 3D array.

    Parameters:
    - data (ndarray): The input 3D array.
    - order (int): The order of the local maxima filter. Default is 1.

    Returns:
    - ndarray: The modified 3D array with local maxima set to non-zero values.
    """
    data = data.numpy()
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    data[data <= filtered] = 0
    return data


def find_peaks(voxel: torch.Tensor):
    voxel[voxel < .25] = 0
    voxel = voxel.squeeze().clone()
    peaks = []
    for channel_idx in range(voxel.shape[0]):
        vox_in = voxel[channel_idx]
        peaks_ = local_maxima(vox_in, 1)
        peaks_ = torch.Tensor(peaks_).unsqueeze(0)
        peaks.append(peaks_)
    peaks = torch.concat(peaks, axis=0)
    return peaks


def get_atom_coords(grid: torch.Tensor, rad: float = 0.5, resolution: float = 0.25):
    coords = []
    atoms_channel = []
    radius = []

    for channel_idx in range(grid.shape[0]):
        px, py, pz = torch.where(grid[channel_idx] > 0)
        px, py, pz = px.float(), py.float(), pz.float()
        coords.append(torch.cat([px.unsqueeze(1), py.unsqueeze(1), pz.unsqueeze(1)], axis=1))
        atoms_channel.append(torch.Tensor(px.shape[0]).fill_(channel_idx))
        radius.append(torch.Tensor(px.shape[0]).fill_(rad))
    coords = (torch.cat(coords, 0).unsqueeze(0) - grid.shape[-1] / 2) * resolution 
    if coords.shape[1] == 0:
        return None
    coords = coords - torch.mean(coords, 1)

    return {
        "coords": coords,
        "atoms_channel": torch.cat(atoms_channel, 0).unsqueeze(0),
        "radius": torch.cat(radius, 0).unsqueeze(0),
    }

