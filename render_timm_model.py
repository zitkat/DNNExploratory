#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

import click
from pathlib import Path
import torch
import timm

from util import now,  get_timm_model
from multi_renders import render_model
from settings import load_settings


@click.command(epilog='(c) 2020 T. Zitka, KKY UWB')
@click.argument("model-name")
@click.option("-w", "--model-weights", default="initialized",
              help="Can be 'pretrained', 'initialized' or path to pth file with state dict.")
@click.option("-sv", "--settings-version", type=str, default="Default",
              help="column in settings file to use as settings")
@click.option("--settings-file", type=Path, default=Path("settings.ods"))
@click.option("--output", type=Path, default=Path("data"))
@click.option("--hide-progress", is_flag=True)
def main(model_name : str, model_weights : str, settings_version : str,
         settings_file : Path, output : Path, hide_progress : bool):
    modes = ["neuron", "channel"]

    settings = load_settings(settings_file, settings_version)

    print(f"{now()} Rendering {model_name}: {model_weights}")

    name = model_weights
    if model_weights.endswith(".pth"):
        model_weights = Path(model_weights)
        name = "finetuned"  # model_weights.with_suffix("").name
        model = get_timm_model(model_name, target_size=5)
        net_dict = torch.load(model_weights)
        model.load_state_dict(net_dict)
    elif model_weights == "pretrained":
        model = timm.create_model(model_name, pretrained=True)
    elif model_weights == "initialized":
        save_path = output / (model_name + "_init.pth")
        if save_path.exists():
            print(f"Loading existing initialization from {save_path}")
            model = get_timm_model(model_name, target_size=5)
            net_dict = torch.load(save_path)
            model.load_state_dict(net_dict)
        else:
            model = get_timm_model(model_name, pretrained=False, target_size=5)
            torch.save(model.cpu().state_dict(), save_path)
    else:
        print(f"Unknown option for model weights {model_weights}")
        return
    outputs_path = Path(output, name + "_" + model_name)

    layers = open(outputs_path / "layers.list", "r").read().split("\n")

    render_model(model,
                 layers=layers,
                 modes=modes,
                 outputs_path=outputs_path,
                 output_suffix=settings_version,
                 progress=not hide_progress,
                 **settings["render"])


if __name__ == '__main__':
    main()
