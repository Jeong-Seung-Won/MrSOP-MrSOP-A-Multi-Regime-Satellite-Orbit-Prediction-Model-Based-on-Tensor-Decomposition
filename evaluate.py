import argparse, os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import load_config
from utils import AverageMeter, ID_TO_REGIME, compute_metrics_by_regime
from dataset import OrbitWindowDataset, collate
from sampler import build_sampler
import models
from models.base import build_model


def evaluate_model(model, loader, device):
    m = {k: AverageMeter() for k in
         ("mse_norm", "mae_norm", "mse_denorm", "mae_denorm",
          "pos3d_mse", "pos3d_mae",

          "pos3d_mse_norm", "pos3d_mae_norm")}
    reg = {}
    model.eval()
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device); y = b["y"].to(device)
            aux = b["router_aux"].to(device)
            rid = b["regime_id"].to(device) if "regime_id" in b else None
            pred, pred_norm, norm_fn = model(x, router_aux=aux, regime_id=rid, return_norm=True)
            y_norm = norm_fn(y)
            bs = x.size(0)

            m["mse_norm"].update(((pred_norm - y_norm) ** 2).mean().item(), bs)
            m["mae_norm"].update((pred_norm - y_norm).abs().mean().item(), bs)
            m["mse_denorm"].update(((pred - y) ** 2).mean().item(), bs)
            m["mae_denorm"].update((pred - y).abs().mean().item(), bs)


            pos_err = pred[..., :3] - y[..., :3]
            dist = torch.linalg.norm(pos_err, dim=-1)
            m["pos3d_mae"].update(dist.mean().item(), bs)
            m["pos3d_mse"].update((dist ** 2).mean().item(), bs)


            pos_err_n = pred_norm[..., :3] - y_norm[..., :3]
            dist_n = torch.linalg.norm(pos_err_n, dim=-1)
            m["pos3d_mae_norm"].update(dist_n.mean().item(), bs)
            m["pos3d_mse_norm"].update((dist_n ** 2).mean().item(), bs)
            if "regime_id" in b:
                for rid in torch.unique(b["regime_id"]):
                    msk = (b["regime_id"] == rid).to(device)
                    name = ID_TO_REGIME.get(int(rid), str(int(rid)))
                    e = (pred[msk] - y[msk])
                    e_n = (pred_norm[msk] - y_norm[msk])
                    dm = dist[msk]
                    dm_n = dist_n[msk]
                    d = reg.setdefault(name, {
                        "mse": AverageMeter(), "mae": AverageMeter(),
                        "pos3d_mae": AverageMeter(), "pos3d_mse": AverageMeter(),
                        "mse_norm": AverageMeter(), "mae_norm": AverageMeter(),
                        "pos3d_mae_norm": AverageMeter(), "pos3d_mse_norm": AverageMeter()})
                    n = int(msk.sum())
                    d["mse"].update((e ** 2).mean().item(), n)
                    d["mae"].update(e.abs().mean().item(), n)
                    d["pos3d_mae"].update(dm.mean().item(), n)
                    d["pos3d_mse"].update((dm ** 2).mean().item(), n)
                    d["mse_norm"].update((e_n ** 2).mean().item(), n)
                    d["mae_norm"].update(e_n.abs().mean().item(), n)
                    d["pos3d_mae_norm"].update(dm_n.mean().item(), n)
                    d["pos3d_mse_norm"].update((dm_n ** 2).mean().item(), n)
    return ({k: v.avg for k, v in m.items()},
            {k: {kk: vv.avg for kk, vv in v.items()} for k, v in reg.items()})


def main():
    extra = argparse.ArgumentParser(description='Evaluate a MrSOP checkpoint on a balanced in-domain sample.')
    extra.add_argument("--split", default="test", choices=["test", "val", "train"])
    extra.add_argument("--ckpt", default=None, help="Checkpoint path (default: {ckpt_dir}/{exp}_best.pt)")
    extra.add_argument("--num-windows", type=int, default=None, help="Override evaluation sample count")
    extra.add_argument("--config", default="configs/mrsop.yaml")
    extra.add_argument("--set", nargs="*", default=[], metavar="section.key=value")
    known = extra.parse_args()
    cfg = load_config(["--config", known.config, "--set", *known.set])

    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    csv = {"test": cfg.data.test_csv, "val": cfg.data.val_csv,
           "train": cfg.data.train_csv}[known.split]
    ckpt_path = known.ckpt or os.path.join(cfg.train.ckpt_dir, f"{cfg.exp_name}_best.pt")

    print(f"[eval] split={known.split} csv={csv}")
    print(f"[eval] ckpt={ckpt_path}")

    ds = OrbitWindowDataset(csv, cfg.data.look_back, cfg.data.horizon,
                            cfg.data.stride, cfg.data.n_channels, cfg.data.unit)
    print(f"  total windows: {len(ds):,}")


    n_eval = cfg.data.val_windows if known.split != "test" else max(cfg.data.val_windows, 100000)
    if known.num_windows is not None:
        if known.num_windows <= 0:
            raise ValueError("--num-windows must be positive")
        n_eval = known.num_windows
    samp, _ = build_sampler(ds, temperature=0.0, num_samples=n_eval, seed=12345)
    idx = list(iter(samp))
    sub = Subset(ds, idx)
    print(f"  eval windows: {len(idx):,} (regime-balanced, seed=12345)")
    dl = DataLoader(sub, batch_size=cfg.train.batch_size, shuffle=False,
                    num_workers=cfg.data.num_workers, collate_fn=collate, pin_memory=True)

    model = build_model(cfg, initialize_tt=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    print(f"  loaded: epoch={state.get('epoch','?')} "
          f"val_mae_denorm={state.get('val_mae_denorm','?')}")

    m, reg = evaluate_model(model, dl, device)
    print("\n" + "=" * 60)
    print(f"[{known.split}] Overall")
    print("=" * 60)
    print(f"  -- 6-channel metrics (position + velocity) --")
    print(f"  [denorm] MSE = {m['mse_denorm']:.2f} km^2   MAE = {m['mae_denorm']:.2f} km   "
          f"RMSE = {m['mse_denorm']**0.5:.2f} km")
    print(f"  [norm]   MSE = {m['mse_norm']:.4f}   MAE = {m['mae_norm']:.4f}   "
          f"RMSE = {m['mse_norm']**0.5:.4f}")
    print(f"  -- 3D position error (Euclidean position distance) --")
    print(f"  [denorm] MAE = {m['pos3d_mae']:.2f} km   MSE = {m['pos3d_mse']:.2f} km^2   "
          f"RMSE = {m['pos3d_mse']**0.5:.2f} km")
    print(f"  [norm]   MAE = {m['pos3d_mae_norm']:.4f}   MSE = {m['pos3d_mse_norm']:.4f}   "
          f"RMSE = {m['pos3d_mse_norm']**0.5:.4f}")
    print(f"\n  By regime (denorm km):")
    print(f"  {'regime':>6} {'MAE':>10} {'RMSE':>10} {'Pos3D_MAE':>11} {'Pos3D_RMSE':>12}")
    for name in ["LEO", "MEO", "NSO", "GEO"]:
        if name in reg:
            r = reg[name]
            print(f"  {name:>6} {r['mae']:>10.2f} {r['mse']**0.5:>10.2f} "
                  f"{r['pos3d_mae']:>11.2f} {r['pos3d_mse']**0.5:>12.2f}")
    print(f"\n  By regime (norm):")
    print(f"  {'regime':>6} {'MAE':>10} {'RMSE':>10} {'Pos3D_MAE':>11} {'Pos3D_RMSE':>12}")
    for name in ["LEO", "MEO", "NSO", "GEO"]:
        if name in reg:
            r = reg[name]
            print(f"  {name:>6} {r['mae_norm']:>10.4f} {r['mse_norm']**0.5:>10.4f} "
                  f"{r['pos3d_mae_norm']:>11.4f} {r['pos3d_mse_norm']**0.5:>12.4f}")


if __name__ == "__main__":
    main()
