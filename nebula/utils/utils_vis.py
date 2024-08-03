import os
import torch
import numpy as np
import plotly.graph_objects as go
from matplotlib.pylab import plt
import seaborn as sns

from utils.utils import makedir

sns.set_theme()

COLORS = [
    "Greys",
    "Greens",
    "Reds",
    "Blues",
    "Oranges",
    "Mint",
    "Magenta",
    "Purples",
    "Peach",
    [[0, "rgb(0,0,0)"], [1, "rgb(255,191,0)"]],    # Yellow
    [[0, "rgb(0,0,0)"], [1, "rgb(175, 96, 26)"]],  # Brown
]


def visualize_voxel_plotly(voxel, id, dirname="figures/", threshold=0.1, to_png=True, to_html=False):
    voxel = voxel.squeeze()
    voxel[voxel < threshold] = 0
    X, Y, Z = np.mgrid[:voxel.shape[-3], :voxel.shape[-2], :voxel.shape[-1]]

    fig = go.Figure()
    for channel in range(voxel.shape[0]):
        voxel_channel = voxel[channel:channel+1]
        if voxel_channel.sum().item() == 0:
            continue
        fig.add_volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=voxel_channel.flatten(),
            isomin=0.1,
            isomax=.3,  # 0.8,
            opacity=0.1,  # 0.075, # needs to be small to see through all surfaces
            surface_count=17,  # needs to be a large number for good volume rendering
            colorscale=COLORS[channel],
            showscale=False
        )
    if to_html:
        fig.write_html(f"{dirname}/{id}.html")
    if to_png:
        fig.write_image(f"{dirname}/{id}.png")


def visualize_voxel_matplotlib(voxel, id, dirname="figures/", to_png=True, to_html=False):
    ax = plt.figure().add_subplot(projection="3d")
    for channel in range(voxel.shape[0]):
        ax.voxels(voxel[channel], alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(id)

    plt.savefig(f"{dirname}/{id}_all_channels.png", bbox_inches="tight")
    # plt.show()
    plt.close()


def render_generated_samples(samples, dirname, save_voxels=False, to_html=False):
    """Render and save a list of generated samples at a given epoch.

    Args:
        samples (list): list or torch.Tensor voxel samples
        dirname (str): path for dir to save the generated samples
    """
    makedir(dirname)

    # save voxel grids
    if save_voxels:
        torch.save(samples, os.path.join(os.path.split(dirname)[0], "samples.pth"))

    # save png/html
    for i, sample in enumerate(samples):
        sample = sample - sample.min()
        sample = sample / sample.max()
        sample[sample < .1] = 0
        visualize_voxel_plotly(sample, f"{i:02d}", dirname=dirname, to_png=True, to_html=to_html)
