#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from collections import OrderedDict
from typing import TypeVar
from pathlib import Path

import torch
from torch import nn
import numpy as np

from util import get_timm_model


T = TypeVar('T', bound='MappedModel')


class MappedModel(nn.Module):

    def __init__(self, model):
        super(MappedModel, self).__init__()
        self.model = model
        self.layers = build_layers_dict(self.model)

        self.activations = OrderedDict()
        self.record_activations = False
        for name, layer in self.layers.items():
            layer : nn.Module
            layer.register_forward_hook(self._get_activation_hook(name))


    def forward(self, *args, return_activations=True, **kwargs):
        out = self.model.forward(*args, **kwargs)
        if self.record_activations and return_activations:
            return self.activations
        return out

    def train(self: T, mode: bool = True) -> T:
        self.record_activations = not mode
        return super(MappedModel, self).train(mode)


    def _get_activation_hook(self, name):
        def hook(model, input, output):
            if self.record_activations:
                self.activations[name] = output.detach()
        return hook



    def __getitem__(self, item):
        if isinstance(item, slice):
            # TODO return slice of layers as invocable, mapped module
            raise NotImplemented("TODO return slice of layers as invocable, mapped module")
        elif isinstance(item, list):
            # TODO reuturn list of layers
           raise NotImplemented("TODO reuturn list of layers")
        elif isinstance(item, tuple):
            # TODO return neuron weights base on (layer, n) tuple
            raise NotImplemented("TODO return neuron weights base on (layer, n) tuple")
        elif isinstance(item, str):
            return self.layers[item]


def build_layers_dict(module):
    layers = OrderedDict()

    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    continue
                layers["_".join(prefix+[name])] = layer
                get_layers(layer, prefix=prefix+[name])

    get_layers(module)
    return layers


if __name__ == '__main__':
    model = get_timm_model("seresnext50_32x4d", target_size=5)
    net_dict = torch.load("data/seresnext50_32x4d_0_best.pth")
    model.load_state_dict(net_dict)

    mmodel = MappedModel(model).eval().to(0)
    all_layers = list(mmodel.layers.keys())
    rendered_path = Path("data/pretrained_seresnext50_32x4d/npys")
    rendered_layers = list(rendered_path.glob("*.npy"))
    all_conv_layers = list(filter(lambda s: "conv" in s, all_layers))
    rendered_layers = ["_".join(fl.stem.split("_")[1:-1]) for fl in rendered_layers]
    print("All layers", len(all_conv_layers))
    len(rendered_layers)
    print("Rendered layers", len(set(rendered_layers)))
    todo_layers = list(set(all_conv_layers) - set(rendered_layers))
    print("TODO layers", len(todo_layers))
    open(rendered_path.parent / "layers.list", "w").write("\n".join(todo_layers))