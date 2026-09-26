import os, time, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import load_config, save_config
from utils import (set_seed, AverageMeter, get_logger,
                   compute_metrics, compute_metrics_by_regime, ID_TO_REGIME)
from dataset import OrbitWindowDataset, collate
from sampler import build_sampler
import models
from models.base import build_model


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def evaluate(model, loader, device):

    model.eval()
    m = {k: AverageMeter() for k in
         ("mse_norm", "mae_norm", "mse_denorm", "mae_denorm")}
    reg_mse = {}
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
            if "regime_id" in b:
                rm = compute_metrics_by_regime(pred, y, rid)
                for name, mt in rm.items():
                    n_reg = int((rid == next(k for k, v in ID_TO_REGIME.items() if v == name)).sum())
                    reg_mse.setdefault(name, AverageMeter()).update(mt["mse"], n_reg)
    return ({k: v.avg for k, v in m.items()},
            {k: v.avg for k, v in reg_mse.items()})


def main():
    cfg = load_config()
    set_seed(cfg.train.seed)
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    logger = get_logger("orbit_fm", os.path.join(cfg.train.ckpt_dir, f"{cfg.exp_name}.log"))
    save_config(cfg, os.path.join(cfg.train.ckpt_dir, f"{cfg.exp_name}.yaml"))
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    logger.info(f"exp={cfg.exp_name} device={device} model={cfg.model.name}")


    train_ds = OrbitWindowDataset(cfg.data.train_csv, cfg.data.look_back,
                                  cfg.data.horizon, cfg.data.stride,
                                  cfg.data.n_channels, cfg.data.unit)
    val_ds = OrbitWindowDataset(cfg.data.val_csv, cfg.data.look_back,
                                cfg.data.horizon, cfg.data.stride,
                                cfg.data.n_channels, cfg.data.unit)
    logger.info(f"train windows: {len(train_ds):,} | val windows: {len(val_ds):,}")

    sampler, info = build_sampler(train_ds, cfg.data.sampling_temperature,
                                  num_samples=cfg.data.windows_per_epoch)
    logger.info(f"regime sampling (temp={cfg.data.sampling_temperature}): {info}")

    _pf = {"prefetch_factor": cfg.data.prefetch_factor} if cfg.data.num_workers > 0 else {}
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
                          num_workers=cfg.data.num_workers, collate_fn=collate,
                          pin_memory=torch.device(device).type == "cuda", drop_last=True,
                          persistent_workers=cfg.data.num_workers > 0, **_pf)


    from torch.utils.data import Subset
    val_sampler, _ = build_sampler(val_ds, temperature=0.0,
                                   num_samples=cfg.data.val_windows, seed=12345)
    val_indices = list(iter(val_sampler))
    val_subset = Subset(val_ds, val_indices)
    logger.info(f"Fixed validation subset: {len(val_indices):,} windows (regime-balanced)")
    val_dl = DataLoader(val_subset, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers, collate_fn=collate,
                        pin_memory=torch.device(device).type == "cuda")


    model = build_model(cfg).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model params: total={nparam:,}  trainable={ntrain:,}")

    _decay, _nodecay = [], []
    for _n, _p in model.named_parameters():
        if not _p.requires_grad:
            continue
        (_nodecay if "log_tau" in _n else _decay).append(_p)
    if _nodecay:
        print("[no-decay] %d params (log_tau)" % sum(p.numel() for p in _nodecay))
    opt = torch.optim.AdamW(
        [{"params": _decay, "weight_decay": cfg.train.weight_decay},
         {"params": _nodecay, "weight_decay": 0.0}],
        lr=cfg.train.lr)
    steps_per_epoch = len(train_dl)
    if steps_per_epoch == 0:
        raise ValueError("No training batches: windows_per_epoch must be at least batch_size")
    _accum = int(getattr(cfg.train, "grad_accum", 1) or 1)
    if _accum > 1:
        print("[grad-accum] %d (effective batch size = %d)"
              % (_accum, cfg.train.batch_size * _accum))
    total_steps = math.ceil(steps_per_epoch / _accum) * cfg.train.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, cfg.train.warmup_steps, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and torch.device(device).type == "cuda")
    loss_fn = nn.MSELoss()

    best_val = float("inf"); patience = 0; gstep = 0
    start_epoch = 0
    _last_path = os.path.join(cfg.train.ckpt_dir, cfg.exp_name + "_last.pt")
    if cfg.train.resume and not os.path.exists(_last_path):
        raise FileNotFoundError(f"Cannot resume: {_last_path}")
    if cfg.train.resume:
        _ck = torch.load(_last_path, map_location=device, weights_only=True)
        model.load_state_dict(_ck["model"])
        opt.load_state_dict(_ck["opt"])
        sched.load_state_dict(_ck["sched"])
        scaler.load_state_dict(_ck["scaler"])
        start_epoch = _ck["epoch"] + 1
        best_val = _ck.get("best_val", float("inf"))
        patience = _ck.get("patience", 0)
        gstep = _ck.get("gstep", 0)
        logger.info("[resume] Resuming from epoch %d (best_val=%.2f)" % (start_epoch, best_val))

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train(); t_train = time.time()
        tm = {k: AverageMeter() for k in
              ("loss", "mse_norm", "mae_norm", "mse_denorm", "mae_denorm")}
        for i, b in enumerate(train_dl):
            x = b["x"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            aux = b["router_aux"].to(device, non_blocking=True)
            if i % _accum == 0:
                opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.train.amp and torch.device(device).type == "cuda"):
                rid = b["regime_id"].to(device) if "regime_id" in b else None
                pred, pred_norm, norm_fn = model(x, router_aux=aux, regime_id=rid, return_norm=True)
                y_norm = norm_fn(y)
                loss = loss_fn(pred_norm, y_norm)
                aux_loss = getattr(model, "_last_aux_loss", None)
                if aux_loss is not None:
                    loss = loss + aux_loss
            group_size = min(_accum, steps_per_epoch - (i // _accum) * _accum)
            scaler.scale(loss / group_size).backward()
            if (i + 1) % _accum == 0 or i + 1 == steps_per_epoch:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(opt); scaler.update(); sched.step()
                gstep += 1
            bs = x.size(0)
            tm["loss"].update(loss.item(), bs)
            with torch.no_grad():
                tm["mse_norm"].update(((pred_norm - y_norm) ** 2).mean().item(), bs)
                tm["mae_norm"].update((pred_norm - y_norm).abs().mean().item(), bs)
                tm["mse_denorm"].update(((pred - y) ** 2).mean().item(), bs)
                tm["mae_denorm"].update((pred - y).abs().mean().item(), bs)
            if (i + 1) % cfg.train.log_every == 0:
                logger.info(f"ep{epoch} [{i+1}/{steps_per_epoch}] "
                            f"loss={tm['loss'].avg:.4f} "
                            f"lr={sched.get_last_lr()[0]:.2e}")
        train_time = time.time() - t_train
        logger.info(
            f"[train] ep{epoch} time={train_time:.0f}s | "
            f"MSE(norm)={tm['mse_norm'].avg:.4f} MAE(norm)={tm['mae_norm'].avg:.4f} | "
            f"MSE(denorm)={tm['mse_denorm'].avg:.2f} MAE(denorm)={tm['mae_denorm'].avg:.2f}")


        if (epoch + 1) % cfg.train.val_every == 0:
            t_val = time.time()
            vm, reg = evaluate(model, val_dl, device)
            val_time = time.time() - t_val
            reg_str = " ".join(f"{k}={v:.2f}" for k, v in sorted(reg.items()))
            logger.info(
                f"[val]   ep{epoch} time={val_time:.0f}s | "
                f"MSE(norm)={vm['mse_norm']:.4f} MAE(norm)={vm['mae_norm']:.4f} | "
                f"MSE(denorm)={vm['mse_denorm']:.2f} MAE(denorm)={vm['mae_denorm']:.2f}")
            logger.info(f"        regime MSE(denorm): {reg_str}")
            val_score = vm["mae_denorm"]
            if val_score < best_val:
                best_val = val_score; patience = 0
                ckpt = os.path.join(cfg.train.ckpt_dir, f"{cfg.exp_name}_best.pt")
                torch.save({"model": model.state_dict(), "exp": cfg.exp_name,
                            "epoch": epoch, "val_mse_norm": vm["mse_norm"],
                            "val_mae_denorm": vm["mae_denorm"]}, ckpt)
                logger.info(f"  -> best saved (val MAE denorm={val_score:.2f} km): {ckpt}")
            else:
                patience += 1
                if cfg.train.early_stop_patience > 0 and patience >= cfg.train.early_stop_patience:
                    logger.info(f"early stop at epoch {epoch} (best MAE denorm={best_val:.2f} km)")
                    break
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "gstep": gstep, "patience": patience,
                        "best_val": min(best_val, vm["mae_denorm"]),
                        "exp": cfg.exp_name},
                       os.path.join(cfg.train.ckpt_dir, cfg.exp_name + "_last.pt"))
    logger.info(f"done. best val MAE(denorm)={best_val:.2f} km")


if __name__ == "__main__":
    main()
