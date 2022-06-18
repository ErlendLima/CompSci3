from warnings import warn

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from posterior import Posterior
from stubs import Axes, Array, Figure
from typing import Any, TypeAlias, Optional
from scipy.stats import gaussian_kde

PltKwargs: TypeAlias = Optional[dict[Any, Any]]


def marginal_plot(posterior: Posterior, fig: Figure | None = None,
                  bins: int = 20,
                  cmap: Any | None = None,
                  plot_map: bool = False,
                  plot_mean: bool = False,
                  histkw: PltKwargs = None,
                  scatterkw: PltKwargs = None,
                  contourkw: PltKwargs = None,
                  mapkw: PltKwargs = None,
                  meankw: PltKwargs = None,
                  ) -> (Figure, dict[Any, Axes]):
    histkw = {} if histkw is None else histkw
    scatterkw = {} if scatterkw is None else scatterkw
    contourkw = {} if contourkw is None else contourkw
    mapkw = {} if mapkw is None else mapkw
    meankw = {} if meankw is None else meankw

    # Let `bins` and `cmap` set the default for all plots
    histkw = {'bins': bins} | histkw
    contourkw = {'bins': bins} | contourkw
    if cmap is not None:
        contourkw = {'cmap': cmap} | contourkw
        scatterkw = {'cmap': cmap} | scatterkw
    mapkw = {'color': 'C1'} | mapkw
    meankw = {'color': 'C2'} | meankw

    N = posterior.data.model.num_parameters
    samples = posterior.samples
    model_params, noise_params, likelihood = posterior.data.model.split_samples(samples)
    params = np.hstack([model_params, noise_params])
    labels= posterior.data.model.parameters()
    mosaic = [['' for i in range(N)] for i in range(N)]
    for i in range(N):
        for j in range(i+1):
            mosaic[i][j] = f'{i}{j}'
    if fig is None:
        fig = plt.figure()
    axd = fig.subplot_mosaic(mosaic, empty_sentinel='')
    # Share x and y. This is mind-breaking to get right
    for i in range(N):
        for j in range(1,i+1):
            if i == j:
                continue
            axd[f'{i}{j}'].get_shared_x_axes().join(axd[f'{i}{j}'], axd[f'{i-1}{j}'])
            axd[f'{i}{j}'].get_shared_y_axes().join(axd[f'{i}{j}'], axd[f'{i}{j-1}'])

    for i in range(N):
        # Diagonal historgrams
        ax = axd[f'{i}{i}']
        plot_hist(ax, params[:, i], **histkw)
        ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
        ax.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True)
        if plot_mean:
            plot_lines(ax, posterior.mean[i], **meankw)
        if plot_map:
            plot_lines(ax, posterior.MAP[i], **mapkw)
        # Scatter 2D and contour
        for j in range(i):
            ax = axd[f'{i}{j}']
            scatter(ax, params[:, j], params[:, i],
                    **scatterkw)
            contour(ax, params[:, j], params[:, i],
                    **contourkw)
            if plot_mean:
                plot_lines(ax, posterior.mean[j], posterior.mean[i], **meankw)
            if plot_map:
                plot_lines(ax, posterior.MAP[j], posterior.MAP[i], **mapkw)
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='y', left=False, labelleft=False)
            if j == 0:
                ax.tick_params(axis='y', left=True, labelleft=True)
                ax.set_ylabel(labels[i])
            if i == N-1:
                ax.tick_params(axis='x', bottom=True, labelbottom=True)
                ax.set_xlabel(labels[j])

    #fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig, axd


def plot_hist(ax: Axes, param: Array, **kwargs) -> None:
    kwargs = {'bins': 20} | kwargs
    ax.hist(param, **kwargs)


def scatter(ax: Axes, p1: Array, p2: Array, **kwargs) -> None:
    kwargs = {'s': 1, 'rasterized': True} | kwargs
    xy = np.vstack([p1, p2])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = p1[idx], p2[idx], z[idx]
    ax.scatter(p1, p2, c=z, zorder=-1, **kwargs)


def contour(ax: Axes, x: Array, y: Array,
            bins: int = 20,
            weights: Array | None = None,
            levels: Array | None = None,
            smooth: float | None = 1,
            **kwargs) -> None:
    # A lot of this code is "inspired" from corner.py's core.hist2d
    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    H, X, Y = np.histogram2d(
        x.flatten(),
        y.flatten(),
        bins=bins,
        weights=weights
    )
    if smooth is not None:
        H = gaussian_filter(H, smooth)

    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        warn("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
            ]
    )
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
            ]
    )

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2
    )
    # Plot the base fill to hide the densest data points.
    if False:
        ax.contourf(
            X2,
            Y2,
            H2.T,
            [V.min(), H.max()],
            cmap=white_cmap,
            antialiased=False,
        )
    ax.contour(X1, Y1, H.T, V, **kwargs)
    #ax.contour(X2, Y2, H2.T, V, **kwargs)


def plot_lines(ax: Axes, x: float, y: float | None = None, **kwargs) -> None:
    ax.axvline(x=x, **kwargs)
    if y is not None:
        ax.axhline(y=y, **kwargs)
