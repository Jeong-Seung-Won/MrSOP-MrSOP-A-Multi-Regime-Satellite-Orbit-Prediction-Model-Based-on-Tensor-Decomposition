import random, logging, sys
import numpy as np
import torch
import torch.nn as nn

MU = 398600.4418
REGIME_IDS = {"LEO": 0, "MEO": 1, "NSO": 2, "GEO": 3}
ID_TO_REGIME = {v: k for k, v in REGIME_IDS.items()}


class RevIN(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        self._mean = None
        self._std = None

    def fit(self, x_lookback: torch.Tensor):
        self._mean = x_lookback.mean(dim=1, keepdim=True)
        self._std = x_lookback.std(dim=1, keepdim=True, unbiased=False)
        return self

    def normalize(self, x: torch.Tensor):
        x = (x - self._mean) / (self._std + self.eps)
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def denormalize(self, x: torch.Tensor):
        if self.affine:
            x = (x - self.beta) / (self.gamma + 1e-8)
        return x * (self._std + self.eps) + self._mean


def compute_router_aux(x_lookback: torch.Tensor) -> torch.Tensor:

    last = x_lookback[:, -1, :]
    r = torch.linalg.norm(last[:, :3], dim=1)
    v = torch.linalg.norm(last[:, 3:6], dim=1)
    a = 1.0 / (2.0 / r - v * v / MU)
    h = torch.cross(last[:, :3], last[:, 3:6], dim=1)
    e_vec = torch.cross(last[:, 3:6], h, dim=1) / MU - last[:, :3] / r[:, None]
    e = torch.linalg.norm(e_vec, dim=1)
    log_a = torch.log10(torch.clamp(a, min=1.0))
    return torch.stack([log_a, e], dim=1)


def compute_router_aux_np(window: np.ndarray) -> np.ndarray:

    last = window[-1]
    r = np.linalg.norm(last[:3]); v = np.linalg.norm(last[3:6])
    a = 1.0 / (2.0 / r - v * v / MU)
    h = np.cross(last[:3], last[3:6])
    e = np.linalg.norm(np.cross(last[3:6], h) / MU - last[:3] / r)
    return np.array([np.log10(max(a, 1.0)), e], dtype=np.float32)


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor):

    err = pred - target
    mse = (err ** 2).mean().item()
    mae = err.abs().mean().item()
    rmse = mse ** 0.5

    pos_rmse = ((err[..., :3] ** 2).mean().item()) ** 0.5
    vel_rmse = ((err[..., 3:6] ** 2).mean().item()) ** 0.5
    return dict(mse=mse, mae=mae, rmse=rmse, pos_rmse=pos_rmse, vel_rmse=vel_rmse)


@torch.no_grad()
def compute_metrics_by_regime(pred, target, regime_ids):

    out = {}
    for rid in torch.unique(regime_ids):
        m = regime_ids == rid
        name = ID_TO_REGIME.get(int(rid), str(int(rid)))
        out[name] = compute_metrics(pred[m], target[m])
    return out


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = 0.0; self.n = 0
    def update(self, val, k=1): self.sum += val * k; self.n += k
    @property
    def avg(self): return self.sum / max(self.n, 1)


def get_logger(name="orbit_fm", logfile=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if logfile:
        fh = logging.FileHandler(logfile); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger
