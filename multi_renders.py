#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"
import numpy as np
import matplotlib.pyplot as plt

from lucent.optvis import render, param
from lucent.modelzoo.util import get_model_layers

from settings import transforms
from util import ncobj, batch_indices, now, ensured_path
from vizualizations import show_fvs


def render_layer(model, layer, idcs, mode="neuron",
                 batch_size=6,
                 image_size=(50,),
                 optimizer=None,
                 transforms=transforms,
                 thresholds=(512,),
                 verbose=False,
                 preprocess=True,
                 progress=True,
                 show_image=False,
                 save_image=False,
                 image_name=None,
                 show_inline=False,
                 fixed_image_size=None):
    res_list = []
    for indcs_batch in batch_indices(idcs, batch_size=batch_size):
        batch_param_f = lambda: param.image(*image_size, batch=len(indcs_batch))
        obj = sum([ncobj(mode, layer, n, b) for b, n in enumerate(indcs_batch)])
        res_list += render.render_vis(model, obj, batch_param_f,
                                     optimizer=optimizer,
                                     transforms=transforms,
                                     thresholds=thresholds,
                                     verbose=verbose,
                                     preprocess=preprocess,
                                     progress=progress,
                                     show_image=show_image,
                                     save_image=save_image,
                                     image_name=image_name,
                                     show_inline=show_inline,
                                     fixed_image_size=fixed_image_size,
                                     # desc=f"{layer} | units: {indcs_batch}"
                                      )
    return np.concatenate(res_list, axis=0)


def render_model(model, layers, idcs=None, modes=("neuron",),
                 outputs_path=None, output_suffix="",
                 batch_size=6,
                 image_size=(50,),
                 optimizer=None,
                 transforms=transforms,
                 thresholds=(512,),
                 verbose=False,
                 preprocess=True,
                 progress=True,
                 show_image=False,
                 save_image=False,
                 image_name=None,
                 show_inline=False,
                 fixed_image_size=None):
    # TODO use mapped model
    model = model.to(0).eval()
    all_layers = get_model_layers(model)

    # REFACTOR move to function
    if layers == "all":
        print("Warning: rendering ALL layers, this might be caused by default "
              "value and will take really long!")
        selected_layers = all_layers
    elif callable(layers):
        selected_layers = [ln for ln in all_layers if layers(ln)]
    elif isinstance(layers, list):
        selected_layers = layers
    elif isinstance(layers, str):
        selected_layers = [ln for ln in all_layers if layers in ln] + ["fc"]
    else:
        raise ValueError("Unsupported specification of layers to render.")

    for layer in selected_layers:
        for mode in modes:
            print(f"\n\n{now()} Starting layer {layer} - {mode}s\n")
            ns = range(64 if layer != "fc" else 5)  # TODO use idcs parameter

            output_npys_path = ensured_path((outputs_path / "npys") / (mode + "s_" + layer + "_" + output_suffix))
            output_fig_path = ensured_path((outputs_path / "figs") / (mode + "s_" + layer + "_" + output_suffix + ".png"))
            if not output_npys_path.with_suffix(".npy").exists():
                res = render_layer(model, layer, ns,
                                   mode=mode,
                                   batch_size=batch_size,
                                   image_size=image_size,
                                   optimizer=optimizer,
                                   transforms=transforms,
                                   thresholds=thresholds,
                                   verbose=verbose,
                                   preprocess=preprocess,
                                   progress=progress,
                                   show_image=show_image,
                                   save_image=save_image,
                                   image_name=image_name,
                                   show_inline=show_inline,
                                   fixed_image_size=fixed_image_size
                                   )
                np.save(output_npys_path, res)
                f, a = show_fvs(res, ns, max_cols=8)
                f.savefig(output_fig_path)
                plt.close()
            else:
                print(f"{output_npys_path} already exists.")