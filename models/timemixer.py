import torch.nn as nn
import torch.nn.functional as F

from utils import RevIN, compute_router_aux
from .base import BaseBackbone
from .moe import MoEFeedForward


def moving_avg_decomp(z, kernel):

    padding = kernel // 2
    transposed = z.transpose(1, 2)
    padded = F.pad(transposed, (padding, padding), mode="replicate")
    trend = F.avg_pool1d(padded, kernel, stride=1)
    if trend.shape[-1] != transposed.shape[-1]:
        trend = trend[..., :transposed.shape[-1]]
    trend = trend.transpose(1, 2)
    return z - trend, trend


class SeasonMixing(nn.Module):


    def __init__(self, lengths, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lengths[i], lengths[i + 1]), nn.GELU(),
                nn.Linear(lengths[i + 1], lengths[i + 1]), nn.Dropout(dropout),
            )
            for i in range(len(lengths) - 1)
        ])

    def forward(self, season):
        current = season[0]
        out = [current]
        for i in range(len(season) - 1):
            current = season[i + 1] + self.layers[i](current)
            out.append(current)
        return out


class TrendMixing(nn.Module):


    def __init__(self, lengths, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lengths[i + 1], lengths[i]), nn.GELU(),
                nn.Linear(lengths[i], lengths[i]), nn.Dropout(dropout),
            )
            for i in reversed(range(len(lengths) - 1))
        ])

    def forward(self, trend):
        reversed_trend = trend[::-1]
        current = reversed_trend[0]
        out = [current]
        for i in range(len(reversed_trend) - 1):
            current = reversed_trend[i + 1] + self.layers[i](current)
            out.append(current)
        return out[::-1]


class PDMBlock(nn.Module):
    def __init__(self, cfg, lengths, *, initialize_tt=True):
        super().__init__()
        model = cfg.model
        self.kernel = model.decomp_kernel
        self.season_mix = SeasonMixing(lengths, model.dropout)
        self.trend_mix = TrendMixing(lengths, model.dropout)
        self.out_cross = MoEFeedForward(
            model.d_model, model.d_ff, n_experts=model.n_experts,
            top_k=model.moe_top_k, dropout=model.dropout,
            tt_rank=model.tt_rank, router_tau=model.router_tau,
            initialize_tt=initialize_tt,
        )

    def forward(self, x_list, router_aux):
        seasons, trends = [], []
        for x in x_list:
            season, trend = moving_avg_decomp(x, self.kernel)
            seasons.append(season.transpose(1, 2))
            trends.append(trend.transpose(1, 2))
        mixed_seasons = self.season_mix(seasons)
        mixed_trends = self.trend_mix(trends)
        out = []
        aux_total = None
        for original, season, trend in zip(x_list, mixed_seasons, mixed_trends):
            mixed = (season + trend).transpose(1, 2)
            expert_out, aux = self.out_cross(mixed, router_aux=router_aux)
            out.append(original + expert_out)
            aux_total = aux if aux_total is None else aux_total + aux
        return out, aux_total


class MrSOP(BaseBackbone):


    def __init__(self, cfg, *, initialize_tt=True):
        super().__init__()
        model, data = cfg.model, cfg.data
        self.L, self.H, self.C = data.look_back, data.horizon, data.n_channels
        self.revin = RevIN(self.C, eps=model.revin_eps, affine=model.revin_affine)
        self.dsw = model.down_sampling_window
        self.n_down = model.n_scales - 1
        self.lens = [self.L // (self.dsw ** i) for i in range(self.n_down + 1)]
        if not self.lens or min(self.lens) < 1:
            raise ValueError("look_back is too short for the requested scales")
        self.e_layers = model.n_layers
        self.embed = nn.Linear(1, model.d_model)
        self.pdm = nn.ModuleList([
            PDMBlock(cfg, self.lens, initialize_tt=initialize_tt)
            for _ in range(self.e_layers)
        ])
        self.predictors = nn.ModuleList([nn.Linear(length, self.H) for length in self.lens])
        self.projection = nn.Linear(model.d_model, 1)

    def forward(self, x, router_aux=None, regime_id=None, return_norm=False):
        if x.ndim != 3 or x.shape[1:] != (self.L, self.C):
            raise ValueError(f"Expected x with shape [batch, {self.L}, {self.C}]")
        batch = x.shape[0]
        if router_aux is None:
            router_aux = compute_router_aux(x)
        if router_aux.shape != (batch, 2):
            raise ValueError("router_aux must have shape [batch, 2]")
        self.revin.fit(x)
        normalized = self.revin.normalize(x)

        scales = [normalized]
        current = normalized.transpose(1, 2)
        for _ in range(self.n_down):
            current = F.avg_pool1d(current, self.dsw)
            scales.append(current.transpose(1, 2))

        embeddings = []
        for scale in scales:
            time = scale.shape[1]
            channel_independent = scale.permute(0, 2, 1).reshape(batch * self.C, time, 1)
            embeddings.append(self.embed(channel_independent))

        expanded_aux = router_aux.repeat_interleave(self.C, dim=0)
        aux_loss = x.new_zeros(())
        for block in self.pdm:
            embeddings, aux = block(embeddings, router_aux=expanded_aux)
            if aux is not None:
                aux_loss = aux_loss + aux

        decoded = 0
        for i, embedding in enumerate(embeddings):
            prediction = self.predictors[i](embedding.transpose(1, 2)).transpose(1, 2)
            decoded = decoded + prediction
        out = self.projection(decoded)
        out = out.reshape(batch, self.C, self.H).permute(0, 2, 1)
        prediction = self.revin.denormalize(out)
        self._last_aux_loss = aux_loss
        if return_norm:
            return prediction, out, self.revin.normalize
        return prediction


OrbitTimeMixer = MrSOP
