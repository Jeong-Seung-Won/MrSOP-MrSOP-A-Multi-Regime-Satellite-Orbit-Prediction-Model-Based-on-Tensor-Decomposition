import argparse
from pathlib import Path

import numpy as np
import torch

from config import load_config
from models import build_model


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def resolve_device(name):
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(name)


def load_model(config_path, checkpoint_path, device="cpu"):

    cfg = load_config(["--config", str(config_path)])
    device = resolve_device(device)
    model = build_model(cfg, initialize_tt=False)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError("checkpoint must be a state dictionary or contain a 'model' dictionary")
    weights = saved.get("model", saved)
    if not isinstance(weights, dict):
        raise ValueError("checkpoint['model'] must be a state dictionary")
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    return model, cfg, device


def validate_inputs(values, look_back):
    if values.ndim not in (2, 3) or values.shape[-2:] != (look_back, 6):
        raise ValueError(f"input must have shape [{look_back}, 6] or [N, {look_back}, 6]")
    if values.size == 0:
        raise ValueError("input cannot be empty")
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("input must contain finite real numbers")


@torch.inference_mode()
def predict(model, values, cfg, device, batch_size=32, unit="km"):

    validate_inputs(values, cfg.data.look_back)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if unit not in ("km", "m"):
        raise ValueError("unit must be km or m")
    single = values.ndim == 2
    if single:
        values = values[None]
    result = np.empty((len(values), cfg.data.horizon, 6), dtype=np.float32)
    scale = 1e-3 if unit == "m" else 1.0
    for start in range(0, len(values), batch_size):
        batch = np.array(values[start:start + batch_size], dtype=np.float32, copy=True)
        batch *= scale
        if not np.isfinite(batch).all():
            raise ValueError("input exceeds the float32 range used by the model")
        forecast = model(torch.from_numpy(batch).to(device))
        if not torch.isfinite(forecast).all():
            raise ValueError("model produced nonfinite predictions; check input states and checkpoint")
        result[start:start + len(batch)] = forecast.cpu().numpy() / scale
    if not np.isfinite(result).all():
        raise ValueError("predictions exceed float32 range in the requested output units")
    return result[0] if single else result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Forecast orbital states from sampled Cartesian input windows.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/mrsop.yaml", help="model YAML configuration")
    parser.add_argument("--checkpoint", required=True, help="trained .pt checkpoint")
    parser.add_argument("--input", required=True, help="sampled Cartesian-state NPY array")
    parser.add_argument("--output", required=True, help="forecast NPY path (same units as input)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--unit", choices=("km", "m"), default="km",
                        help="input and output position units; velocities use the same distance unit/s")
    args = parser.parse_args(argv)
    try:
        model, cfg, device = load_model(args.config, args.checkpoint, args.device)
        values = np.load(args.input, allow_pickle=False, mmap_mode="r")
        output = predict(model, values, cfg, device, args.batch_size, args.unit)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.save(stream, output, allow_pickle=False)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Saved {output.shape} forecasts to {destination} ({args.unit}, {args.unit}/s).")


if __name__ == "__main__":
    main()
