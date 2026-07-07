import random
import pickle

import numpy as np
import matplotlib.pyplot as plt

from mnist1d.transform import transform


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def to_pickle(thing, path):
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=3)


def from_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


def plot_signals(xs, t, labels=None, args=None, ratio=2.6, do_transform=False, dark_mode=False, zoom=1):
    rows, cols = 1, 10
    fig = plt.figure(figsize=[cols * 1.5, rows * 1.5 * ratio], dpi=60)
    for r in range(rows):
        for c in range(cols):
            ix = r * cols + c
            x, t_i = xs[ix], t
            ax = plt.subplot(rows, cols, ix + 1)

            if do_transform:
                assert args is not None, "Need an args object in order to do transforms"
                x, t_i = transform(x, t_i, args)
            if dark_mode:
                plt.plot(x, t_i, 'wo', linewidth=6)
                ax.set_facecolor('k')
            else:
                plt.plot(x, t_i, 'k-', linewidth=2)
            if labels is not None:
                plt.title("label=" + str(labels[ix]), fontsize=22)

            plt.xlim(-zoom, zoom)
            plt.ylim(-zoom, zoom)
            plt.gca().invert_yaxis()
            plt.xticks([], [])
            plt.yticks([], [])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()
    return fig
