import os, csv as csvmod
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import compute_router_aux_np, REGIME_IDS

CH_ORDER = [0, 1, 2, 3, 4, 5]


class OrbitWindowDataset(Dataset):
    def __init__(self, csv_path, look_back=192, horizon=96, stride=15,
                 n_channels=6, unit="km", return_regime=True):
        if n_channels != 6 or unit not in ("km", "m"):
            raise ValueError("Expected six channels and unit km or m")
        manifest_dir = Path(csv_path).resolve().parent
        self.L = look_back
        self.H = horizon
        self.stride = stride
        self.C = n_channels
        self.unit_scale = 1.0 if unit == "km" else 1e-3
        self.return_regime = return_regime
        self.span = (look_back + horizon - 1) * stride + 1


        self.paths = []
        self.seg_start = []
        self.seg_end = []
        self.regime = []
        counts = []
        with open(csv_path, newline="") as fp:
            rd = csvmod.DictReader(fp)
            for row in rd:
                s, e = int(row["seg_start"]), int(row["seg_end"])
                seg_len = e - s
                if seg_len < self.span:
                    continue
                nwin = (seg_len - self.span) // stride + 1
                path = Path(row["path"])
                if not path.is_absolute():
                    path = manifest_dir / path
                if row["regime"].upper() not in REGIME_IDS:
                    raise ValueError(f"Unknown regime: {row['regime']}")
                if s < 0 or e <= s:
                    raise ValueError("Expected 0 <= seg_start < seg_end")
                self.paths.append(str(path))
                self.seg_start.append(s)
                self.seg_end.append(e)
                self.regime.append(REGIME_IDS.get(row["regime"].upper(), -1))
                counts.append(nwin)

        if not counts:
            raise RuntimeError(f"{csv_path}: No segments are long enough to form a window (span={self.span})")

        self.seg_start = np.asarray(self.seg_start, dtype=np.int64)
        self.seg_end = np.asarray(self.seg_end, dtype=np.int64)
        self.regime = np.asarray(self.regime, dtype=np.int8)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.cum = np.cumsum(self.counts)
        self.total = int(self.cum[-1])

    def __len__(self):
        return self.total

    def _resolve(self, gidx):

        fid = int(np.searchsorted(self.cum, gidx, side="right"))
        prev = int(self.cum[fid - 1]) if fid > 0 else 0
        local = gidx - prev
        return fid, local

    def regime_of_window(self, gidx):

        fid = int(np.searchsorted(self.cum, gidx, side="right"))
        return int(self.regime[fid])

    def __getitem__(self, gidx):
        fid, local = self._resolve(gidx)
        start = int(self.seg_start[fid]) + local * self.stride
        idx = start + np.arange(self.L + self.H) * self.stride
        arr = np.load(self.paths[fid], mmap_mode="r", allow_pickle=False)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.ndim != 2 or arr.shape[-1] != 6 or self.seg_end[fid] > len(arr):
            raise ValueError(f"Invalid data shape/segment: {self.paths[fid]}")
        window = np.asarray(arr[idx][:, CH_ORDER], dtype=np.float32) * self.unit_scale

        if not np.isfinite(window).all():
            raise ValueError(f"Non-finite orbit states: {self.paths[fid]}")
        x = window[:self.L]
        y = window[self.L:]
        aux = compute_router_aux_np(x)

        out = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "router_aux": torch.from_numpy(aux),
        }
        if self.return_regime:
            out["regime_id"] = int(self.regime[fid])
        return out


def collate(batch):
    out = {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "router_aux": torch.stack([b["router_aux"] for b in batch]),
    }
    if "regime_id" in batch[0]:
        out["regime_id"] = torch.tensor([b["regime_id"] for b in batch], dtype=torch.long)
    return out
