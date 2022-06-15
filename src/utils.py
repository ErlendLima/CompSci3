from stubs import Axes, Figure
import matplotlib.pyplot as plt
from typing import Optional
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from pathlib import Path


def newfig(N: int = 1,
           height: Optional[float] = None,
           plot3d: bool = False,
           nrows=1,
           ncols=1,
           sharex=False,
           sharey=False,
           addax=True,
           **kwargs):
    fig_width, fig_height = half_fig()
    if height is not None:
        fig_height *= height
    fig_size = [N * ncols * fig_width, N * nrows * fig_height]
    fig = plt.figure(figsize=(fig_size[0], fig_size[1]), **kwargs)
    if addax:
        if plot3d:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.subplots(ncols=ncols,
                              nrows=nrows,
                              sharex=sharex,
                              sharey=sharey)
        return fig, ax
    else:
        return fig


def half_fig(n=1, r=1):
    fig_width_pt = 467.42 / 2  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = n * fig_width_pt * inches_per_pt  # width in inches
    fig_height = r * fig_width * golden_mean  # height in inches
    return fig_width, fig_height


def full_fig(n=1, r=1):
    fig_width_pt = 467.42  # Beamer half column
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = n * fig_width_pt * inches_per_pt  # width in inches
    fig_height = r * fig_width * golden_mean  # height in inches
    return fig_width, fig_height


def savefig(fig,
            path,
            transparent=True,
            dpi=196,
            fig_path=Path('../figures/')):
    fig.savefig(fig_path / path,
                dpi=dpi,
                transparent=transparent,
                bbox_inches='tight')

def maybe_set_xlabel(ax: Axes, label: str):
    if not ax.xaxis.get_label()._label:
        ax.set_xlabel(label)

def maybe_set_ylabel(ax: Axes, label: str):
    if not ax.xaxis.get_label()._label:
        ax.set_ylabel(label)

def maybe_set_title(ax: Axes, title: str):
    if not ax.get_title():
        ax.set_title(title)

def maybe_set_suptitle(fig: Figure, title: str):
    if not fig._suptitle:
        fig.suptitle(title)

Axes.maybe_set_xlabel = maybe_set_xlabel
Axes.maybe_set_ylabel = maybe_set_ylabel
