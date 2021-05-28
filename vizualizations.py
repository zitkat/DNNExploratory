#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"
import matplotlib.pyplot as plt
import numpy as np


def show_fvs(fv_array, ns, max_cols=4):

    N = len(ns)
    rows = max(int(np.ceil(N / max_cols)), 1)
    cols = min(N, max_cols)
    print(rows, cols)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5*max_cols, 5*rows), tight_layout=True)
    fig.subplots_adjust(wspace=0, hspace=1)
    for i, (n, ax) in enumerate(zip(ns, axs.flatten())):
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(n)
        ax.imshow(fv_array[i, ...], interpolation=None)
    return fig, axs