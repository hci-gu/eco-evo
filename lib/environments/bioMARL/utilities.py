import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from PIL import Image


def get_rnd_gen(seed=None):
    return np.random.default_rng(seed)


def rnd_init_h(
    grid_shape,
    n_groups,
    seed=None,
    gaussian=False,
    single_cell=False,
    min_threshold=0.0,
    fg_init_k=None,
    sig=0.1,
    round_to_int=True,
    n_peaks=3,
):
    rs = get_rnd_gen(seed)
    if single_cell:
        h = np.zeros((n_groups, grid_shape[0], grid_shape[1]))
        rx = rs.choice(grid_shape[0], n_groups, replace=True)
        ry = rs.choice(grid_shape[1], n_groups, replace=True)
        for i in range(n_groups):
            h[i, rx[i], ry[i]] = 1
    elif gaussian:
        h_list = []
        for i in range(n_groups):
            h = np.zeros(grid_shape)
            length = h.shape[0]
            scale = length * sig
            indx = np.meshgrid(np.arange(length), np.arange(length))
            locsxy = rs.uniform(0, length, (2, n_peaks))
            for i in range(n_peaks):
                # print(locsxy[:,i])
                h += scipy.stats.norm.pdf(
                    indx[0], loc=locsxy[0, i], scale=scale
                ) * scipy.stats.norm.pdf(indx[1], loc=locsxy[1, i], scale=scale)
            h_list.append(h)

        h = np.array(h_list)
    else:
        h = rs.random((n_groups, grid_shape[0], grid_shape[1]))

    h[h < min_threshold] = 0
    if fg_init_k is not None:
        h = np.einsum("sxy, s -> sxy", h, 1 / np.max(h, axis=(1, 2)))
        h = np.einsum("sxy, s -> sxy", h, fg_init_k)
        if round_to_int:
            h = np.round(h).astype(int)
    return h


def plot_distributions(
    h,
    titles=None,
    show=True,
    outfile=None,
    dpi=250,
    cmap="YlGn",
    vmin=None,
    vmax=None,
    zero_to_nan=True,
):
    h_tmp = h + 0.0
    if zero_to_nan:
        h_tmp[h == 0] = np.nan
    fig = plt.figure(figsize=(16, 4))
    for i in range(h.shape[0]):
        fig.add_subplot(1, h.shape[0], i + 1)
        sns.heatmap(h_tmp[i], cmap=cmap, vmin=vmin, vmax=vmax)
        if titles is not None:
            plt.gca().set_title(titles[i])

    if show:
        plt.show()

    if outfile is not None:
        plt.savefig(outfile, dpi=dpi)
        plt.close()


def plot_distributions_energy(
    h,
    e,
    title_tag="",
    show=True,
    outfile=None,
    dpi=250,
    cmaps=None,
    vmin=None,
    vmax=None,
    zero_to_nan=True,
    map=None,
    mask=None,
):
    if cmaps is None:
        cmaps = ["YlGn", "Reds"]

    h_tmp = h + 0.0
    if zero_to_nan:
        h_tmp[h == 0] = np.nan
    e_tmp = e + 0.0
    if zero_to_nan:
        e_tmp[e == 0] = np.nan

    if map is not None and mask is not None:
        assert map.shape[:2] == mask.shape, "Map and mask must have the same shape."
        # Compute padding.
        h_grid_shape = h.shape[1:]
        padding = (h_grid_shape[0] - map.shape[0], h_grid_shape[1] - map.shape[1])
        if padding[0] != padding[1]:
            raise ValueError("Padding must be the same in both dimensions.")
        if padding[0] < 0:
            raise ValueError("Padding must be positive.")
        if padding[0] % 2 != 0:
            raise ValueError("Padding must be the same on all sides.")
        padding = padding[0] // 2
        h_tmp = h_tmp[:, padding:-padding, padding:-padding]
        e_tmp = e_tmp[:, padding:-padding, padding:-padding]

    pops = np.sum(h, axis=(1, 2))
    titles_h = ["FG: %s (%s) %s:" % (i, pops[i], title_tag) for i in range(len(pops))]

    ene = np.nanmean(e_tmp, axis=(1, 2))
    titles_e = [
        "FG: %s (%s) %s:" % (i, np.round(ene[i], 1), title_tag) for i in range(len(ene))
    ]

    fig = plt.figure(figsize=(16, 6.5))
    fig.tight_layout()

    # Allow for a vmax per FG.
    vmax = [None for _ in range(h.shape[0])] if vmax is None else vmax
    vmin = [None for _ in range(h.shape[0])] if vmin is None else vmin
    # for i in range(h.shape[0]):
    for i, (_h, _vmax, _vmin) in enumerate(zip(h_tmp, vmax, vmin)):
        fig.add_subplot(2, h.shape[0], i + 1)
        sns.heatmap(
            _h,
            cmap=cmaps[0],
            vmin=_vmin,
            vmax=_vmax,
            xticklabels=False,
            yticklabels=False,
            mask=mask,
        )
        if map is not None:
            plt.imshow(map)
        plt.gca().set_title(titles_h[i])

    for j in range(e.shape[0]):
        fig.add_subplot(2, e.shape[0], i + j + 2)
        sns.heatmap(
            e_tmp[j],
            cmap=cmaps[1],
            vmin=0,
            vmax=100,
            xticklabels=False,
            yticklabels=False,
            mask=mask,
        )
        if map is not None:
            plt.imshow(map)
        plt.gca().set_title(titles_e[j])

    if show:
        plt.show()

    if outfile is not None:
        plt.savefig(outfile, dpi=dpi)
        plt.close()


def plot_history(h, log_vars=None, show=True, outfile=None, dpi=250):
    if log_vars is None:
        log_vars = ["pop_sizes"]
    fig = plt.figure(figsize=(16, 4))
    keys = list(h.keys())
    for i in range(len(keys)):
        fig.add_subplot(1, len(keys), i + 1)
        if keys[i] in log_vars:
            plt.yscale("log")
        if i == 0:
            labels = [f"FG{j}" for j in range(len(h[keys[i]][0]))]
            plt.plot(h[keys[i]], label=labels)
        else:
            plt.plot(h[keys[i]])
        plt.gca().set_title(keys[i])

    fig.legend(loc="upper right", ncol=1, bbox_to_anchor=(1, 0.9))

    if show:
        plt.show()

    if outfile is not None:
        plt.savefig(outfile, dpi=dpi)
        plt.close()


def sort_human(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split("([-+]?[0-9]*\.?[0-9]*)", key)]
    l.sort(key=alphanum)
    return l


def sort_step_plots(l):
    f0 = []
    for f in l:
        f0.append(int(f.split("_")[1].split(".")[0]))

    indx_ord = np.unique(f0, return_index=True)[1]
    l = np.array(l)[indx_ord]
    return l


def make_gif(wd, tag=None, outfile=None, duration=100, loop=0):
    folder = os.path.join(wd, tag)
    files = sort_step_plots(glob.glob(folder))
    if outfile is None:
        outfile = "summary.gif"
    frames = [Image.open(image) for image in files]
    frame_one = frames[0]
    frame_one.save(
        outfile,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=loop,
    )


def debug_plot(h, save_file=None):
    # Create a figure with a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Loop through each image in h_2 and plot it
    for i in range(4):
        # Get the current image's min and max
        img_min, img_max = h[i].min(), h[i].max()

        # Display the image with individual vmin and vmax
        im = axes[i].imshow(h[i], cmap="gray", vmin=img_min, vmax=img_max)
        axes[i].set_title(f"Image {i+1} (min: {img_min:.2f}, max: {img_max:.2f})")

        # Add a colorbar for each subplot
        cbar = fig.colorbar(im, ax=axes[i])
        cbar.set_label("Pixel Intensity")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, format="svg")
    plt.show()
