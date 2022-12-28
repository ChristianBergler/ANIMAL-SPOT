"""
Module: summary.py
Authors: Christian Bergler, Hendrik Schroeter
GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 27.12.2022
"""

import os
import librosa
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.ticker as mticker

import torch

from typing import Tuple
from PIL import Image, ImageDraw
from torchvision.utils import make_grid
from visualization.utils import spec2img

"""
Prepare given image data for tensorboard visualization
"""
def prepare_img(img, num_images=4, file_names=None):
    with torch.no_grad():
        if img.shape[0] == 0:
            raise ValueError("`img` must include at least 1 image.")

        if num_images < img.shape[0]:
            tmp = img[:num_images]
        else:
            tmp = img
        tmp = spec2img(tmp)

        if file_names is not None:
            tmp = tmp.permute(0, 3, 2, 1)
            for i in range(tmp.shape[0]):
                try:
                    pil = Image.fromarray(tmp[i].numpy(), mode="RGB")
                    draw = ImageDraw.Draw(pil)
                    draw.text(
                        (2, 2),
                        os.path.basename(file_names[i]),
                        (255, 255, 255),
                    )
                    np_pil = np.asarray(pil).copy()
                    tmp[i] = torch.as_tensor(np_pil)
                except TypeError:
                    pass
            tmp = tmp.permute(0, 3, 1, 2)

        tmp = make_grid(tmp, nrow=1)
        return tmp.numpy()

"""
Prepare audio data for a given input audio file
"""
def prepare_audio(file_names, sr=44100, data_dir=None, num_audios=4) -> torch.Tensor:
    with torch.no_grad():
        out = []
        for i in range(min(num_audios, len(file_names))):
            file_name = file_names[i]
            if data_dir is not None:
                file_name = os.path.join(data_dir, file_name)
                audio, _ = librosa.load(file_name, sr)
                out.append(torch.from_numpy(audio))
        out = torch.cat(out)
        return out.reshape((1, -1))

"""
Plot ROC curve based on TPR/FPR
"""
def roc_fig(tpr, fpr, auc):
    fig = plt.figure()
    plt.plot(fpr, tpr, label="AUC: {}".format(auc))
    plt.legend(markerscale=0)
    plt.title("ROC curve")
    return fig

"""
Plot Confusion Matrix
"""
def confusion_matrix_fig(confusion, label_str=None, numbering=True):
    if isinstance(confusion, torch.Tensor):
        confusion = confusion.numpy()
    if label_str is None:
        label_str = [str(i) for i in range(confusion.shape[0])]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion, cmap="hot_r")
    fig.colorbar(cax)

    tick_size = list(range(0, confusion.shape[0], 1))

    ax.set_xticks(np.array(tick_size))  # just get and reset whatever you already have
    ax.set_xticklabels(label_str, rotation=90)

    ax.set_yticks(np.array(tick_size))  # just get an
    ax.set_yticklabels(label_str)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

    if numbering:
        for (i, j), z in np.ndenumerate(confusion):
            ax.text(j, i, '{:0.1f}'.format(z), size='smaller', weight='bold', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    return fig



"""
Plot Spectrogram
"""
def plot_spectrogram(spectrogram,
                     output_filepath=None,
                     sr: int = 44100,
                     hop_length: int = 441,
                     fmin: int = 50,
                     fmax: int = 12500,
                     title: str = "spectrogram",
                     log=False,
                     show=True,
                     axes=None,
                     ax_title=None,
                     **kwargs
                     ):
    kwargs.setdefault("cmap", plt.cm.get_cmap("viridis"))
    kwargs.setdefault("rasterized", True)
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.squeeze().cpu().numpy()
    spectrogram = spectrogram.T
    figsize: Tuple[int, int] = (5, 10)
    figure = plt.figure(figsize=figsize)
    figure.suptitle(title)
    if log:
        f = np.logspace(np.log2(fmin), np.log2(fmax), num=spectrogram.shape[0], base=2)
    else:
        f = np.linspace(fmin, fmax, num=spectrogram.shape[0])
    t = np.arange(0, spectrogram.shape[1]) * hop_length / sr
    if axes is None:
        axes = plt.gca()
    if ax_title is not None:
        axes.set_title(ax_title)
    img = axes.pcolormesh(t, f, spectrogram, shading="auto", **kwargs)
    figure.colorbar(img, ax=axes)
    axes.set_xlim(t[0], t[-1])
    axes.set_ylim(f[0], f[-1])
    if log:
        axes.set_yscale("symlog", basey=2)
    yaxis = axes.yaxis
    yaxis.set_major_formatter(tick.ScalarFormatter())
    xaxis = axes.xaxis
    xaxis.set_label_text("time [s]")
    if show:
        plt.show()
    plt.savefig(output_filepath)
    plt.close("all")