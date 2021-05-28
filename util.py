#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"
from pathlib import Path

import time

import timm
from lucent.optvis import objectives
from torch import nn as nn


def make_path(*pathargs, isdir=False, **pathkwargs):
    new_path = Path(*pathargs, **pathkwargs)
    return ensured_path(new_path, isdir=isdir)


def ensured_path(path: Path, isdir=False):
    if isdir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def now():
    """
    :return: date and time as YYYY-mm-dd-hh-MM
    """
    return time.strftime("%Y-%m-%d-%H-%M")


def ncobj(t, layer, n, batch=0):
    obj_constructor = getattr(objectives, t)
    return obj_constructor(layer, n , batch=batch)


def batch_indices(indcs, batch_size):
    indcs = list(indcs)
    start = 0
    end = batch_size
    while start < len(indcs):
        yield indcs[start:end]
        start = end
        end += batch_size


def get_timm_model(architecture_name, target_size, pretrained=False):
    net = timm.create_model(architecture_name, pretrained=pretrained)
    net_cfg = net.default_cfg
    last_layer = net_cfg['classifier']
    num_ftrs = getattr(net, last_layer).in_features
    setattr(net, last_layer, nn.Linear(num_ftrs, target_size))
    return net

