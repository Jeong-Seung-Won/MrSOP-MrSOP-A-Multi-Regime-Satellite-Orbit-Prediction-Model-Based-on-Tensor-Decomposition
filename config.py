from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import argparse
import yaml


@dataclass
class DataConfig:
    train_csv: str = "data/train.csv"
    val_csv: str = "data/val.csv"
    test_csv: str = "data/test.csv"
    unit: str = "km"
    num_workers: int = 4
    prefetch_factor: int = 4
    look_back: int = 192
    horizon: int = 96
    stride: int = 15
    n_channels: int = 6
    sampling_temperature: float = 0.5
    windows_per_epoch: int = 500000
    val_windows: int = 50000


@dataclass
class ModelConfig:
    name: str = "mrsop"
    d_model: int = 512
    d_ff: int = 1024
    n_layers: int = 2
    n_scales: int = 3
    down_sampling_window: int = 2
    decomp_kernel: int = 25
    dropout: float = 0.1
    revin_eps: float = 1e-5
    revin_affine: bool = False
    n_experts: int = 16
    moe_top_k: int = 2
    tt_rank: int = 32
    router_tau: float = 0.15


@dataclass
class TrainConfig:
    batch_size: int = 64
    grad_accum: int = 1
    epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    warmup_steps: int = 500
    seed: int = 42
    device: str = "cuda"
    amp: bool = True
    log_every: int = 100
    val_every: int = 1
    ckpt_dir: str = "checkpoints"
    early_stop_patience: int = 0
    resume: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    exp_name: str = "mrsop"


def _coerce(value, expected):
    if isinstance(expected, bool):
        if str(value).lower() not in ("true", "false", "1", "0", "yes", "no"):
            raise ValueError(f"Invalid boolean: {value}")
        return str(value).lower() in ("true", "1", "yes")
    return type(expected)(value)


def _from_dict(data):
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping")
    cfg = Config()
    for section, values in data.items():
        if section == "exp_name":
            cfg.exp_name = str(values)
            continue
        if section not in ("data", "model", "train"):
            raise KeyError(f"Unknown config key: {section}")
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a mapping")
        obj = getattr(cfg, section)
        valid = {f.name for f in fields(obj)}
        for key, value in values.items():
            if key not in valid:
                raise KeyError(f"Unknown config key: {section}.{key}")
            setattr(obj, key, _coerce(value, getattr(obj, key)))
    return cfg


def validate_config(cfg):
    if cfg.model.name != "mrsop":
        raise ValueError("This release contains only model.name=mrsop")
    if cfg.data.n_channels != 6:
        raise ValueError("MrSOP expects x,y,z,vx,vy,vz (six channels)")
    if cfg.data.unit not in ("km", "m"):
        raise ValueError("data.unit must be km or m")
    for obj, keys in ((cfg.data, ("look_back", "horizon", "stride", "windows_per_epoch", "val_windows")),
                      (cfg.model, ("d_model", "d_ff", "n_layers", "n_scales", "down_sampling_window", "decomp_kernel", "n_experts", "moe_top_k", "tt_rank")),
                      (cfg.train, ("batch_size", "grad_accum", "epochs", "log_every", "val_every"))):
        for key in keys:
            if getattr(obj, key) <= 0:
                raise ValueError(f"{key} must be positive")
    if cfg.data.num_workers < 0 or cfg.train.warmup_steps < 0:
        raise ValueError("num_workers and warmup_steps must be nonnegative")
    if not 1 <= cfg.model.moe_top_k <= cfg.model.n_experts:
        raise ValueError("moe_top_k must be between 1 and n_experts")
    if cfg.model.router_tau <= 0 or not 0 <= cfg.model.dropout < 1:
        raise ValueError("Invalid router_tau or dropout")
    if cfg.data.look_back // cfg.model.down_sampling_window ** (cfg.model.n_scales - 1) < 1:
        raise ValueError("look_back is too short for the configured scales")
    return cfg


def load_config(argv=None):
    parser = argparse.ArgumentParser(description="MrSOP configuration")
    parser.add_argument("--config", default="configs/mrsop.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="section.key=value")
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as stream:
        cfg = _from_dict(yaml.safe_load(stream) or {})
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Expected section.key=value: {item}")
        key, value = item.split("=", 1)
        if key == "exp_name":
            cfg.exp_name = value
        else:
            section, key = key.split(".", 1)
            if section not in ("data", "model", "train"):
                raise KeyError(section)
            obj = getattr(cfg, section)
            setattr(obj, key, _coerce(value, getattr(obj, key)))
    return validate_config(cfg)


def save_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
