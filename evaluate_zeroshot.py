import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from inference import load_model, positive_int

CHANNELS = ("x", "y", "z", "vx", "vy", "vz")
NATIVE_SECONDS = {"starlink": 60, "gnss": 900, "beidou": 900}


def read_ephemeris(path, native_seconds):

    states, epochs = [], []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(("epoch_ms",) + CHANNELS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            try:
                epoch = int(row["epoch_ms"])
                state = [float(row[channel]) for channel in CHANNELS]
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{row_number}: invalid timestamp or state") from exc
            if not np.isfinite(state).all():
                raise ValueError(f"{path}:{row_number}: nonfinite state")
            epochs.append(epoch)
            states.append(state)
    if not states:
        raise ValueError(f"{path}: no ephemeris rows")
    states = np.asarray(states, dtype=np.float32)
    if not np.isfinite(states).all():
        raise ValueError(f"{path}: state exceeds float32 range")
    epochs = np.asarray(epochs, dtype=np.int64)
    delta = np.diff(epochs)
    expected = native_seconds * 1000
    if np.any(delta <= 0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    if np.any(delta % expected != 0):
        raise ValueError(f"{path}: timestamps are inconsistent with {native_seconds}-second native cadence")
    boundaries = np.concatenate(([0], np.flatnonzero(delta != expected) + 1, [len(states)]))
    return states, boundaries


def window_batches(states, boundaries, look_back, horizon, batch_size, resample_stride=1):

    inputs, targets = [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        segment = states[left:right:resample_stride]
        for start in range(0, len(segment) - look_back - horizon + 1, horizon):
            inputs.append(segment[start:start + look_back])
            targets.append(segment[start + look_back:start + look_back + horizon])
            if len(inputs) == batch_size:
                yield np.stack(inputs), np.stack(targets)
                inputs, targets = [], []
    if inputs:
        yield np.stack(inputs), np.stack(targets)


@torch.inference_mode()
def evaluate(model, cfg, device, files, source, batch_size, cadence="recorded"):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cadence not in ("recorded", "native"):
        raise ValueError("cadence must be recorded or native")
    native_seconds = NATIVE_SECONDS[source]
    target_seconds = cfg.data.stride * 60 if cadence == "recorded" else native_seconds
    if target_seconds < native_seconds or target_seconds % native_seconds != 0:
        raise ValueError("Recorded sampling interval must be a positive multiple of source cadence")
    resample_stride = target_seconds // native_seconds
    sums = dict(mae6=0.0, mse6=0.0, pos3d_mae=0.0, pos3d_mse=0.0,
                mae_norm=0.0, mse_norm=0.0, pos3d_mae_norm=0.0, pos3d_mse_norm=0.0)
    total_windows, used_files, gap_count = 0, 0, 0
    for path in files:
        states, boundaries = read_ephemeris(path, native_seconds)
        gap_count += len(boundaries) - 2
        file_windows = 0
        for x, y in window_batches(states, boundaries,
                                    cfg.data.look_back, cfg.data.horizon, batch_size, resample_stride):
            xb, yb = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            pred, pred_norm, norm_fn = model(xb, return_norm=True)
            target_norm = norm_fn(yb)
            if not torch.isfinite(pred).all() or not torch.isfinite(pred_norm).all():
                raise ValueError(f"{path}: model produced nonfinite predictions")
            errors = (pred - yb).double()
            normalized_errors = (pred_norm - target_norm).double()
            distance = torch.linalg.vector_norm(errors[..., :3], dim=-1)
            normalized_distance = torch.linalg.vector_norm(normalized_errors[..., :3], dim=-1)
            metrics = dict(mae6=errors.abs().mean(), mse6=errors.square().mean(),
                           pos3d_mae=distance.mean(), pos3d_mse=distance.square().mean(),
                           mae_norm=normalized_errors.abs().mean(), mse_norm=normalized_errors.square().mean(),
                           pos3d_mae_norm=normalized_distance.mean(),
                           pos3d_mse_norm=normalized_distance.square().mean())
            for name, value in metrics.items():
                if not torch.isfinite(value):
                    raise ValueError(f"{path}: nonfinite evaluation metric {name}")
                sums[name] += value.item() * len(x)
            file_windows += len(x)
        total_windows += file_windows
        used_files += int(file_windows > 0)
    if total_windows == 0:
        raise ValueError("no complete evaluation windows; check file lengths and cadence")
    result = {name: value / total_windows for name, value in sums.items()}
    result.update(source=source, cadence_protocol=cadence, n_windows=total_windows, n_files=len(files),
                  n_files_with_windows=used_files, timestamp_gaps=gap_count,
                  sample_interval_seconds=target_seconds,
                  look_back=cfg.data.look_back, horizon=cfg.data.horizon,
                  window_stride=cfg.data.horizon)
    result["pos3d_rmse"] = result["pos3d_mse"] ** 0.5
    result["pos3d_rmse_norm"] = result["pos3d_mse_norm"] ** 0.5
    result["units"] = {"positions": "km", "velocities": "km/s",
                       "pos3d_mae": "km", "pos3d_rmse": "km", "pos3d_mse": "km^2",
                       "mae6": "mixed position/velocity units", "normalized_metrics": "dimensionless"}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate MrSOP on Starlink, GNSS, or BeiDou CSV ephemerides.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/mrsop.yaml")
    parser.add_argument("--checkpoint", required=True, help="trained .pt checkpoint")
    parser.add_argument("--data-dir", required=True, help="directory containing ephemeris CSV files")
    parser.add_argument("--source", required=True, choices=tuple(NATIVE_SECONDS))
    parser.add_argument("--cadence", choices=("recorded", "native"), default="recorded",
                        help="recorded: original stride*60 seconds; native: source cadence without downsampling")
    parser.add_argument("--pattern", default="*.csv", help="recursive file pattern (default: *.csv)")
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", help="optional JSON results path")
    args = parser.parse_args(argv)
    try:
        directory = Path(args.data_dir)
        if not directory.is_dir():
            raise ValueError(f"not a directory: {directory}")
        files = sorted(path for path in directory.rglob(args.pattern) if path.is_file())
        if not files:
            raise ValueError(f"no files match {args.pattern} under {directory}")
        model, cfg, device = load_model(args.config, args.checkpoint, args.device)
        result = evaluate(model, cfg, device, files, args.source, args.batch_size, args.cadence)
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"[{result['source']}] windows={result['n_windows']:,}, files={result['n_files_with_windows']}/{result['n_files']}")
    print(f"6ch MAE={result['mae6']:.6g} (mixed position/velocity units); normalized MAE={result['mae_norm']:.6g}")
    print(f"Pos3D MAE={result['pos3d_mae']:.6g} km; RMSE={result['pos3d_rmse']:.6g} km")
    print(f"Pos3D normalized MAE={result['pos3d_mae_norm']:.6g}; RMSE={result['pos3d_rmse_norm']:.6g}")
    print(f"Protocol: {result['cadence_protocol']}")
    print(f"Timestamp gaps split: {result['timestamp_gaps']}; cadence: {result['sample_interval_seconds']} s")


if __name__ == "__main__":
    main()
