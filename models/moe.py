import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class TTExperts(nn.Module):


    def __init__(self, n_experts, d_model, d_ff, rank=32, dropout=0.0,
                 *, initialize_tt=True):
        super().__init__()
        self.E, self.d_model, self.d_ff = n_experts, d_model, d_ff
        self.dropout = nn.Dropout(dropout)
        if min(n_experts, d_model, d_ff, rank) < 1:
            raise ValueError("Expert dimensions and TT rank must be positive")

        def make_cores(shape):
            if initialize_tt:
                from tensorly.decomposition import matrix_product_state

                weight = torch.empty(*shape)
                for expert in range(shape[0]):
                    nn.init.kaiming_uniform_(weight[expert], a=5 ** 0.5)
                ranks = [1, min(rank, shape[0]), rank, 1]
                factors = matrix_product_state(weight.numpy(), ranks)
                return nn.ParameterList([
                    nn.Parameter(torch.tensor(factor, dtype=torch.float32))
                    for factor in factors
                ])

            r1 = min(rank, shape[0], shape[1] * shape[2])
            r2 = min(rank, r1 * shape[1], shape[2])
            return nn.ParameterList([
                nn.Parameter(torch.zeros(1, shape[0], r1)),
                nn.Parameter(torch.zeros(r1, shape[1], r2)),
                nn.Parameter(torch.zeros(r2, shape[2], 1)),
            ])

        self.w1_cores = make_cores((n_experts, d_ff, d_model))
        self.w2_cores = make_cores((n_experts, d_model, d_ff))
        self.b1 = nn.Parameter(torch.zeros(n_experts, d_ff))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))

    @staticmethod
    def _reconstruct(cores):
        g1, g2, g3 = cores
        weight = torch.tensordot(g1, g2, dims=([2], [0]))
        weight = torch.tensordot(weight, g3, dims=([3], [0]))
        return weight.squeeze(0).squeeze(-1)

    def _expert_fwd(self, x, w1, w2, b1, b2):
        hidden = self.dropout(F.gelu(x @ w1.t() + b1))
        return hidden @ w2.t() + b2

    def forward(self, x, gate):
        w1 = self._reconstruct(self.w1_cores)
        w2 = self._reconstruct(self.w2_cores)
        batch, time, width = x.shape
        out = torch.zeros(batch, time, width, device=x.device, dtype=x.dtype)
        used = (gate > 0).any(dim=0)
        for expert in range(self.E):
            if not bool(used[expert]):
                continue
            if self.training:
                value = checkpoint(
                    self._expert_fwd, x, w1[expert], w2[expert],
                    self.b1[expert], self.b2[expert], use_reentrant=False,
                )
            else:
                value = self._expert_fwd(
                    x, w1[expert], w2[expert], self.b1[expert], self.b2[expert],
                )
            out = out + gate[:, expert, None, None] * value
        return out


class MoEFeedForward(nn.Module):


    def __init__(self, d_model, d_ff, n_experts=16, top_k=2, dropout=0.0,
                 tt_rank=32, router_tau=0.15, *, initialize_tt=True):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        if router_tau <= 0:
            raise ValueError("router_tau must be positive")
        self.n_experts = n_experts
        self.top_k = top_k
        self.tt_experts = TTExperts(
            n_experts, d_model, d_ff, rank=tt_rank, dropout=dropout,
            initialize_tt=initialize_tt,
        )
        self.register_buffer("aux_mean", torch.tensor([4.20, 0.02]))
        self.register_buffer("aux_std", torch.tensor([0.30, 0.05]))
        prototypes = torch.zeros(n_experts, 2)
        prototypes[:, 0] = torch.linspace(-1.6, 1.6, n_experts)
        prototypes[:, 1] = 0.15 * torch.sin(torch.arange(n_experts).float())
        self.prototypes = nn.Parameter(prototypes)
        self.register_buffer("tau", torch.tensor(float(router_tau)))

    def forward(self, x, router_aux):
        z = (router_aux - self.aux_mean) / self.aux_std
        distance = ((z.unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2).sum(-1)
        gate = F.softmax(-distance / self.tau, dim=-1)
        if self.top_k < self.n_experts:
            top_values, top_indices = gate.topk(self.top_k, dim=-1)
            masked = torch.zeros_like(gate).scatter_(-1, top_indices, top_values)
            gate = masked / masked.sum(-1, keepdim=True).clamp(min=1e-9)
        return self.tt_experts(x, gate), x.new_zeros(())
