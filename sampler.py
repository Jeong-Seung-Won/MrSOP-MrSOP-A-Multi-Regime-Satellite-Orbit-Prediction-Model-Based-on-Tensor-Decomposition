import numpy as np
from torch.utils.data import Sampler

from utils import ID_TO_REGIME


class RegimeBalancedSampler(Sampler):
    def __init__(self, dataset, temperature=0.5, num_samples=None, seed=0):
        self.cum = dataset.cum
        self.total = int(dataset.cum[-1])
        self.num_samples = int(num_samples) if num_samples else self.total
        self.seed = seed
        self.epoch = 0

        regime = dataset.regime
        counts = dataset.counts

        uniq = np.unique(regime)
        ranges = {int(r): [] for r in uniq}
        n_r = {int(r): 0 for r in uniq}
        prev = 0
        for fi in range(len(counts)):
            c = int(counts[fi]); rid = int(regime[fi])
            if c > 0:
                ranges[rid].append((prev, prev + c))
                n_r[rid] += c
            prev += c

        self.rids = sorted(n_r)
        self.n_r = n_r


        self.regime_starts = {}
        self.regime_cum = {}
        for r in self.rids:
            self.regime_starts[r] = np.array([s for s, e in ranges[r]], dtype=np.int64)
            self.regime_cum[r] = np.cumsum([e - s for s, e in ranges[r]], dtype=np.int64)

        raw = np.array([n_r[r] ** temperature for r in self.rids], dtype=np.float64)
        self.p = raw / raw.sum()
        self.info = {ID_TO_REGIME.get(r, str(r)):
                     dict(n_windows=n_r[r], target_frac=round(float(self.p[i]), 4))
                     for i, r in enumerate(self.rids)}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        reg_choices = rng.choice(len(self.rids), size=self.num_samples, p=self.p)
        out = np.empty(self.num_samples, dtype=np.int64)
        for ri, r in enumerate(self.rids):
            mask = reg_choices == ri
            k = int(mask.sum())
            if k == 0:
                continue
            local = rng.choice(self.n_r[r], size=k, replace=True)
            ends = self.regime_cum[r]
            segment = np.searchsorted(ends, local, side="right")
            previous = np.where(segment > 0, ends[np.maximum(segment - 1, 0)], 0)
            out[mask] = self.regime_starts[r][segment] + local - previous
        rng.shuffle(out)
        return iter(out.tolist())


def build_sampler(dataset, temperature=0.5, num_samples=None, seed=0):
    sampler = RegimeBalancedSampler(dataset, temperature, num_samples, seed)
    return sampler, sampler.info
