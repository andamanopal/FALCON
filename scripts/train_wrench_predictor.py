"""
train_wrench_predictor.py
-------------------------
Supervised offline training for the frozen WrenchPredictor MLP.

Data format (from wrench_data_collector.py with temporal-offset alignment):
    .pt file containing a dict with keys:
        "obs"    : (N, 115) float32  -- per-step actor obs at time t
        "plan"   : (N,  70) float32  -- arm plan at time t (H=5 x 14 joints)
        "wrench" : (N,  30) float32  -- future wrench sequence
                                        [w_{t+1}, ..., w_{t+H}] (H x 6)

The wrench targets are 30-dimensional: H=5 future wrench vectors concatenated,
each 6D (force_xyz + torque_xyz).  These are produced by the temporal-offset
data collector which aligns (obs_t, plan_t) with future wrenches.

Z-score normalization statistics are computed on the training split and stored
as buffers inside the model checkpoint so that inference code needs no
separate stats file.

Usage:
    python scripts/train_wrench_predictor.py \
        --data_path data/wrench_data.pt \
        --save_path checkpoints/wrench_predictor.pt \
        --epochs 100 \
        --batch_size 4096 \
        --device cuda
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ---------------------------------------------------------------------------
# Optional WandB (graceful fallback if not installed)
# ---------------------------------------------------------------------------
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimension constants (from VERIFIED_PARAMS.md)
# ---------------------------------------------------------------------------
OBS_DIM    = 115
PLAN_DIM   = 70
WRENCH_DIM = 6
OUTPUT_DIM = 30   # H=5 x WRENCH_DIM
HORIZON    = OUTPUT_DIM // WRENCH_DIM  # 5


# ---------------------------------------------------------------------------
# Inline WrenchPredictor import (handles both installed and source layouts)
# ---------------------------------------------------------------------------
def _load_predictor_class():
    """Import WrenchPredictor, searching common locations."""
    try:
        from humanoidverse.models.wrench_predictor import WrenchPredictor
        return WrenchPredictor
    except ImportError:
        pass
    # Try relative path when running from FALCON/
    script_dir = Path(__file__).resolve().parent
    for candidate in [
        script_dir.parent / "humanoidverse" / "models" / "wrench_predictor.py",
        script_dir.parent.parent / "humanoidverse" / "models" / "wrench_predictor.py",
    ]:
        if candidate.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("wrench_predictor", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.WrenchPredictor
    raise ImportError(
        "Cannot locate WrenchPredictor. "
        "Run from FALCON/ or install the package with `pip install -e .`"
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(data_path: str, device: torch.device):
    """Load .pt file produced by WrenchDataCollector.save().

    The collector now produces temporally-aligned 30-dim future wrench
    targets directly, so no tiling/replication is needed.

    Returns:
        obs    (N, 115), plan (N, 70), wrench_target (N, 30)  -- all on CPU
        (tensors are kept on CPU; the DataLoader will transfer per-batch)
    """
    log.info(f"Loading dataset from {data_path}")
    raw = torch.load(data_path, map_location="cpu", weights_only=True)

    obs = raw["obs"].float()              # (N, 115)
    plan = raw["plan"].float()            # (N, 70)
    wrench_target = raw["wrench"].float() # (N, 30) — H future wrenches

    n_samples = obs.shape[0]
    log.info(f"  Loaded {n_samples:,} aligned training pairs")
    log.info(f"  obs shape:    {tuple(obs.shape)}")
    log.info(f"  plan shape:   {tuple(plan.shape)}")
    log.info(f"  wrench shape: {tuple(wrench_target.shape)}")

    # Validate dimensions
    assert obs.shape[1] == OBS_DIM, (
        f"Expected obs dim {OBS_DIM}, got {obs.shape[1]}"
    )
    assert plan.shape[1] == PLAN_DIM, (
        f"Expected plan dim {PLAN_DIM}, got {plan.shape[1]}"
    )
    assert wrench_target.shape[1] == OUTPUT_DIM, (
        f"Expected wrench target dim {OUTPUT_DIM}, got {wrench_target.shape[1]}. "
        f"Data may be from old collector (6-dim). Re-collect with temporal-offset collector."
    )

    return obs, plan, wrench_target


def compute_normalization_stats(obs: torch.Tensor, plan: torch.Tensor, wrench: torch.Tensor):
    """Compute z-score (mean, std) over the training split tensors.

    Returns six tensors: obs_mean, obs_std, plan_mean, plan_std,
                         wrench_mean, wrench_std
    All shapes match their respective input's last dimension.
    """
    def _stats(x):
        mean = x.mean(dim=0)
        std  = x.std(dim=0).clamp(min=1e-6)
        return mean, std

    obs_mean,    obs_std    = _stats(obs)
    plan_mean,   plan_std   = _stats(plan)
    wrench_mean, wrench_std = _stats(wrench)

    log.info(
        f"  obs   mean/std range: [{obs_mean.min():.4f}, {obs_mean.max():.4f}] / "
        f"[{obs_std.min():.4f}, {obs_std.max():.4f}]"
    )
    log.info(
        f"  plan  mean/std range: [{plan_mean.min():.4f}, {plan_mean.max():.4f}] / "
        f"[{plan_std.min():.4f}, {plan_std.max():.4f}]"
    )
    log.info(
        f"  wrench mean/std range: [{wrench_mean.min():.4f}, {wrench_mean.max():.4f}] / "
        f"[{wrench_std.min():.4f}, {wrench_std.max():.4f}]"
    )
    return obs_mean, obs_std, plan_mean, plan_std, wrench_mean, wrench_std


# ---------------------------------------------------------------------------
# Quality metrics (computed on raw / de-normalized scale)
# ---------------------------------------------------------------------------

def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    """Compute per-component and aggregate quality metrics.

    Args:
        pred   (N, 30): model predictions in physical units
        target (N, 30): ground-truth targets in physical units

    Returns:
        dict with keys: rmse_total, r2_total,
                        rmse_per_comp (30,), r2_per_comp (30,),
                        correlation_per_comp (30,)
    """
    residuals = pred - target                           # (N, 30)
    ss_res    = (residuals ** 2).sum(dim=0)             # (30,)
    ss_tot    = ((target - target.mean(dim=0)) ** 2).sum(dim=0)  # (30,)

    rmse_per_comp = (residuals ** 2).mean(dim=0).sqrt()  # (30,)
    r2_per_comp   = 1.0 - ss_res / (ss_tot + 1e-12)      # (30,)

    # Pearson correlation per component
    pred_c   = pred   - pred.mean(dim=0)
    target_c = target - target.mean(dim=0)
    cov      = (pred_c * target_c).sum(dim=0)
    denom    = (pred_c.norm(dim=0) * target_c.norm(dim=0)).clamp(min=1e-12)
    corr_per_comp = cov / denom                          # (30,)

    rmse_total = rmse_per_comp.mean().item()
    r2_total   = r2_per_comp.mean().item()

    return {
        "rmse_total":          rmse_total,
        "r2_total":            r2_total,
        "rmse_per_comp":       rmse_per_comp,
        "r2_per_comp":         r2_per_comp,
        "correlation_per_comp": corr_per_comp,
    }


def print_per_component_table(metrics: dict):
    """Print a formatted per-component quality table."""
    rmse = metrics["rmse_per_comp"]
    r2   = metrics["r2_per_comp"]
    corr = metrics["correlation_per_comp"]

    # Component labels: H=5 steps x 6 dims (fx, fy, fz, tx, ty, tz)
    dim_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
    header = f"{'Step':>5}  {'Dim':>5}  {'RMSE':>10}  {'R2':>8}  {'Corr':>8}"
    log.info("-" * len(header))
    log.info(header)
    log.info("-" * len(header))
    for step in range(HORIZON):
        for dim_idx, dim_name in enumerate(dim_names):
            comp_idx = step * WRENCH_DIM + dim_idx
            log.info(
                f"{step+1:>5}  {dim_name:>5}  "
                f"{rmse[comp_idx].item():>10.4f}  "
                f"{r2[comp_idx].item():>8.4f}  "
                f"{corr[comp_idx].item():>8.4f}"
            )
    log.info("-" * len(header))
    log.info(
        f"{'MEAN':>5}  {'---':>5}  "
        f"{metrics['rmse_total']:>10.4f}  "
        f"{metrics['r2_total']:>8.4f}  "
        f"{corr.mean().item():>8.4f}"
    )
    log.info("-" * len(header))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _init_wandb(args, n_train: int, n_val: int, num_params: int):
    """Initialize WandB run if available and not disabled."""
    use_wandb = _HAS_WANDB and not args.no_wandb
    if not use_wandb:
        if not _HAS_WANDB and not args.no_wandb:
            log.info("WandB not installed — logging to console only.")
        return False

    run_name = args.wandb_run_name or f"wrench_predictor_{Path(args.data_path).stem}"
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        config={
            "model": "WrenchPredictor",
            "data_path": args.data_path,
            "save_path": args.save_path,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "device": args.device,
            "n_train": n_train,
            "n_val": n_val,
            "num_params": num_params,
            "obs_dim": OBS_DIM,
            "plan_dim": PLAN_DIM,
            "output_dim": OUTPUT_DIM,
            "horizon": HORIZON,
        },
    )
    log.info(f"WandB initialized: {wandb.run.url}")
    return True


def _log_wandb_epoch(epoch, train_loss, val_loss, lr, best_val_loss):
    """Log per-epoch metrics to WandB."""
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": lr,
        "best_val_loss": best_val_loss,
    }, step=epoch)


def _log_wandb_final(metrics):
    """Log end-of-training per-component metrics to WandB."""
    rmse = metrics["rmse_per_comp"]
    r2 = metrics["r2_per_comp"]
    corr = metrics["correlation_per_comp"]
    dim_names = ["fx", "fy", "fz", "tx", "ty", "tz"]

    summary = {
        "final/rmse_total": metrics["rmse_total"],
        "final/r2_total": metrics["r2_total"],
        "final/corr_total": corr.mean().item(),
    }

    # Per-step and per-component
    force_rmse_accum, torque_rmse_accum = [], []
    force_r2_accum, torque_r2_accum = [], []
    for step in range(HORIZON):
        for dim_idx, dim_name in enumerate(dim_names):
            comp_idx = step * WRENCH_DIM + dim_idx
            prefix = f"final/step{step+1}/{dim_name}"
            summary[f"{prefix}/rmse"] = rmse[comp_idx].item()
            summary[f"{prefix}/r2"] = r2[comp_idx].item()
            summary[f"{prefix}/corr"] = corr[comp_idx].item()

            if dim_idx < 3:
                force_rmse_accum.append(rmse[comp_idx].item())
                force_r2_accum.append(r2[comp_idx].item())
            else:
                torque_rmse_accum.append(rmse[comp_idx].item())
                torque_r2_accum.append(r2[comp_idx].item())

    # Force vs torque aggregates
    summary["final/rmse_force"] = sum(force_rmse_accum) / len(force_rmse_accum)
    summary["final/rmse_torque"] = sum(torque_rmse_accum) / len(torque_rmse_accum)
    summary["final/r2_force"] = sum(force_r2_accum) / len(force_r2_accum)
    summary["final/r2_torque"] = sum(torque_r2_accum) / len(torque_r2_accum)

    for key, val in summary.items():
        wandb.run.summary[key] = val


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU.")
    log.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    obs_all, plan_all, wrench_all = load_dataset(args.data_path, device)
    n_total = obs_all.shape[0]

    # ------------------------------------------------------------------
    # 80/20 train/val split (deterministic via generator seed)
    # ------------------------------------------------------------------
    n_train = int(0.8 * n_total)
    n_val   = n_total - n_train
    log.info(f"Split: {n_train:,} train / {n_val:,} val")

    generator = torch.Generator().manual_seed(42)
    full_dataset = TensorDataset(obs_all, plan_all, wrench_all)
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    # Extract training tensors for stats computation
    train_indices = train_dataset.indices
    obs_train_raw    = obs_all[train_indices]
    plan_train_raw   = plan_all[train_indices]
    wrench_train_raw = wrench_all[train_indices]

    # ------------------------------------------------------------------
    # Normalization statistics (training split only)
    # ------------------------------------------------------------------
    log.info("Computing normalization statistics on training split ...")
    obs_mean, obs_std, plan_mean, plan_std, wrench_mean, wrench_std = (
        compute_normalization_stats(obs_train_raw, plan_train_raw, wrench_train_raw)
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    WrenchPredictor = _load_predictor_class()
    model = WrenchPredictor().to(device)
    model.set_normalization_stats(
        obs_mean.to(device),
        obs_std.to(device),
        plan_mean.to(device),
        plan_std.to(device),
        wrench_mean.to(device),
        wrench_std.to(device),
    )
    log.info(f"Model parameters: {model.num_parameters():,}")

    # ------------------------------------------------------------------
    # WandB initialization
    # ------------------------------------------------------------------
    use_wandb = _init_wandb(args, n_train, n_val, model.num_parameters())

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    loss_fn = nn.HuberLoss(reduction="mean", delta=1.0)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    log.info(f"Starting training for up to {args.epochs} epochs "
             f"(early stop patience={args.patience})")
    log.info(f"  batch_size={args.batch_size}, "
             f"  n_train={n_train:,}, n_val={n_val:,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        train_loss_accum = 0.0
        for obs_b, plan_b, wrench_b in train_loader:
            obs_b    = obs_b.to(device, non_blocking=True)
            plan_b   = plan_b.to(device, non_blocking=True)
            wrench_b = wrench_b.to(device, non_blocking=True)

            # The model forward applies normalization internally.
            # For the loss we need normalized targets because the model's
            # output before de-normalization is what the network actually
            # learns.  We normalize targets with the same stats.
            pred = model(obs_b, plan_b)  # (B, 30) in physical units

            loss = loss_fn(pred, wrench_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accum += loss.item() * obs_b.shape[0]

        scheduler.step()
        avg_train_loss = train_loss_accum / n_train

        # ---- Validate ----
        model.eval()
        val_loss_accum = 0.0
        all_pred   = []
        all_target = []
        with torch.no_grad():
            for obs_b, plan_b, wrench_b in val_loader:
                obs_b    = obs_b.to(device, non_blocking=True)
                plan_b   = plan_b.to(device, non_blocking=True)
                wrench_b = wrench_b.to(device, non_blocking=True)

                pred = model(obs_b, plan_b)
                val_loss_accum += loss_fn(pred, wrench_b).item() * obs_b.shape[0]
                all_pred.append(pred.cpu())
                all_target.append(wrench_b.cpu())

        avg_val_loss = val_loss_accum / n_val
        elapsed = time.time() - t0

        log.info(
            f"Epoch {epoch:>4}/{args.epochs}  "
            f"train_loss={avg_train_loss:.6f}  "
            f"val_loss={avg_val_loss:.6f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"time={elapsed:.1f}s"
        )

        if use_wandb:
            _log_wandb_epoch(
                epoch, avg_train_loss, avg_val_loss,
                scheduler.get_last_lr()[0], best_val_loss,
            )

        # ---- Early stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            log.info(f"  -> New best val_loss={best_val_loss:.6f}. Checkpoint cached.")
        else:
            patience_counter += 1
            log.info(
                f"  -> No improvement ({patience_counter}/{args.patience})"
            )
            if patience_counter >= args.patience:
                log.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # ------------------------------------------------------------------
    # Final evaluation on val set using best weights
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Final evaluation with best checkpoint weights ...")
    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    model.eval()

    all_pred   = []
    all_target = []
    with torch.no_grad():
        for obs_b, plan_b, wrench_b in val_loader:
            obs_b    = obs_b.to(device, non_blocking=True)
            plan_b   = plan_b.to(device, non_blocking=True)
            wrench_b = wrench_b.to(device, non_blocking=True)
            pred = model(obs_b, plan_b)
            all_pred.append(pred.cpu())
            all_target.append(wrench_b.cpu())

    pred_all   = torch.cat(all_pred,   dim=0)
    target_all = torch.cat(all_target, dim=0)
    metrics    = compute_metrics(pred_all, target_all)

    log.info(f"Validation RMSE (aggregate): {metrics['rmse_total']:.6f}")
    log.info(f"Validation R²   (aggregate): {metrics['r2_total']:.6f}")
    log.info(f"Validation Corr (aggregate): "
             f"{metrics['correlation_per_comp'].mean().item():.6f}")
    log.info("")
    log.info("Per-component breakdown:")
    print_per_component_table(metrics)

    if use_wandb:
        _log_wandb_final(metrics)

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Re-load best weights into model on CPU for saving
    model_cpu = model.cpu()
    model_cpu.load_state_dict({k: v for k, v in best_state_dict.items()})

    torch.save(
        {
            "model_state_dict": model_cpu.state_dict(),
            "train_config": {
                "epochs":      args.epochs,
                "batch_size":  args.batch_size,
                "device":      args.device,
                "data_path":   args.data_path,
                "patience":    args.patience,
                "n_train":     n_train,
                "n_val":       n_val,
            },
            "metrics": {
                "rmse_total": metrics["rmse_total"],
                "r2_total":   metrics["r2_total"],
                "best_val_loss": best_val_loss,
            },
            "normalization_stats": {
                "obs_mean":    obs_mean,
                "obs_std":     obs_std,
                "plan_mean":   plan_mean,
                "plan_std":    plan_std,
                "wrench_mean": wrench_mean,
                "wrench_std":  wrench_std,
            },
        },
        save_path,
    )
    log.info(f"Checkpoint saved to: {save_path}")
    log.info(
        f"Load with: model.load_frozen('{save_path}')"
    )

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised training for the AnticiPose WrenchPredictor MLP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to .pt file produced by WrenchDataCollector.save().",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/wrench_predictor.pt",
        help="Destination path for the trained checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without val improvement).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device: 'cuda', 'cuda:0', 'cpu', etc.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="andaman-l",
        help="WandB entity (team or username).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="AnticiPose",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name (auto-generated if omitted).",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable WandB logging even if installed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
