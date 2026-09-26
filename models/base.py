import torch.nn as nn


class BaseBackbone(nn.Module):


    def forward(self, x, router_aux=None, regime_id=None, return_norm=False):
        raise NotImplementedError


def build_model(cfg, *, initialize_tt=True):

    from .timemixer import MrSOP

    if cfg.model.name.lower() != "mrsop":
        raise ValueError("This release supports only model.name: mrsop")
    return MrSOP(cfg, initialize_tt=initialize_tt)
