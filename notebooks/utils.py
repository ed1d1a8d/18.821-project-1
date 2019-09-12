import matplotlib
import matplotlib.animation
import matplotlib.pyplot
import numpy as np
from scipy import integrate


def unit_direction(x1, x2):
    """
    Computes the unit vector pointing to x1 to x2

    Parameters
    ----------
    x1: numpy array of shape (d,)
    x2: numpy array of shape (d,)
    """
    return (x2 - x1) / np.linalg.norm(x2 - x1)


def get_dXdt(X):
    """
    Returns dX/dt given X.

    Parameters
    ----------
    X: numpy array of shape (n, d)
        represents the locations of the points
    """
    n = X.shape[0]
    dXdt = np.zeros_like(X)
    for i in range(n):
        dXdt[i] = unit_direction(X[i], X[(i + 1) % n])

    return dXdt


def get_even_pts_on_circle(num_pts, radius=1):
    thetas = np.linspace(start=0, stop=2 * np.pi, num=num_pts, endpoint=False)
    pts = radius * np.array([np.cos(thetas), np.sin(thetas)]).T
    return pts


def sum_follow_distances(X):
    """
    Returns sum of L2 distances between points and their followers.

    Parameters
    ----------
    X: numpy array of shape (n, d)
        represents the locations of the points
    """
    n = X.shape[0]
    ret = 0
    for i in range(n):
        ret += np.linalg.norm(X[(i + 1) % n] - X[i])

    return ret


def get_ani(res, ts, interval=25, xlim=None, ylim=None):
    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 5))
    matplotlib.pyplot.close(
    )  # So we don't get a duplicate plot in the jupyter notebook
    ax.set_aspect('equal')

    n = res.sol(0).shape[0] // 2
    colors = [
        matplotlib.cm.hsv(i) for i in np.linspace(0, 1, n, endpoint=False)
    ]

    def fig_animator(t):
        X = res.sol(t).reshape(-1, 2)

        ax.clear()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.scatter(X[:, 0], X[:, 1], color=colors)

        lines = [(X[i % n], X[(i + 1) % n]) for i in range(n)]
        lc = matplotlib.collections.LineCollection(lines,
                                                   color=colors,
                                                   linewidths=2)
        ax.add_collection(lc)

    ani = matplotlib.animation.FuncAnimation(
        fig,
        fig_animator,
        frames=ts,
        interval=interval,
        repeat=False,
    )

    return ani
