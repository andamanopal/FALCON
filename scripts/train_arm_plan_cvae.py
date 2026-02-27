"""
train_arm_plan_cvae.py
----------------------
Offline CVAE training for the B6 arm plan encoder baseline.

Trains on the same wrench_data.pt as the wrench predictor (uses obs + plan,
ignores wrench).  The CVAE learns to compress the arm plan into a 30-dim
latent conditioned on the current proprioceptive observation.

Loss: MSE(plan_hat, plan_normalized) + beta * KL(q(z|obs,plan) || N(0,I))
Beta warmup: 0.0 for 10 epochs -> linear ramp to 1.0 over 20 epochs -> hold

Usage:
    python scripts/train_arm_plan_cvae.py \
        --data_path data/wrench_data.pt \
        --save_path checkpoints/arm_plan_cvae.pt \
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
# Dimension constants
# ---------------------------------------------------------------------------
OBS_DIM = 115
PLAN_DIM = 70
LATENT_DIM = 30

# ---------------------------------------------------------------------------
# Beta schedule constants
# ---------------------------------------------------------------------------
BETA_WARMUP_START = 10    # epochs of beta=0
BETA_WARMUP_END = 30      # epoch at which beta reaches 1.0
BETA_FINAL = 1.0


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------
def _load_cvae_class():
    """Import ArmPlanCVAE, searching common locations."""
    try:
        from humanoidverse.models.arm_plan_cvae import ArmPlanCVAE
        return ArmPlanCVAE
    except ImportError:
        pass
    script_dir = Path(__file__).resolve().parent
    for candidate in [
        script_dir.parent / "humanoidverse" / "models" / "arm_plan_cvae.py",
        script_dir.parent.parent / "humanoidverse" / "models" / "arm_plan_cvae.py",
    ]:
        if candidate.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("arm_plan_cvae", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.ArmPlanCVAE
    raise ImportError(
        "Cannot locate ArmPlanCVAE. "
        "Run from FALCON/ or install the package with `pip install -e .`"
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(data_path: str):
    """Load .pt wrench data file; use obs + plan only (ignore wrench).

    Returns:
        obs (N, 115), plan (N, 70) — on CPU
    """
    log.info(f"Loading dataset from {data_path}")
    raw = torch.load(data_path, map_location="cpu", weights_only=True)

    obs = raw["obs"].float()
    plan = raw["plan"].float()

    n_samples = obs.shape[0]
    log.info(f"  Loaded {n_samples:,} samples")
    log.info(f"  obs shape:  {tuple(obs.shape)}")
    log.info(f"  plan shape: {tuple(plan.shape)}")

    assert obs.shape[1] == OBS_DIM, (
        f"Expected obs dim {OBS_DIM}, got {obs.shape[1]}"
    )
    assert plan.shape[1] == PLAN_DIM, (
        f"Expected plan dim {PLAN_DIM}, got {plan.shape[1]}"
    )

    return obs, plan


def compute_normalization_stats(obs: torch.Tensor, plan: torch.Tensor):
    """Compute z-score (mean, std) over the training split."""
    def _stats(x):
        mean = x.mean(dim=0)
        std = x.std(dim=0).clamp(min=1e-6)
        return mean, std

    obs_mean, obs_std = _stats(obs)
    plan_mean, plan_std = _stats(plan)

    log.info(
        f"  obs  mean/std range: [{obs_mean.min():.4f}, {obs_mean.max():.4f}] / "
        f"[{obs_std.min():.4f}, {obs_std.max():.4f}]"
    )
    log.info(
        f"  plan mean/std range: [{plan_mean.min():.4f}, {plan_mean.max():.4f}] / "
        f"[{plan_std.min():.4f}, {plan_std.max():.4f}]"
    )
    return obs_mean, obs_std, plan_mean, plan_std


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)), averaged over batch."""
    return -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    )


def get_beta(epoch: int) -> float:
    """Beta warmup schedule: 0 -> linear ramp -> hold at BETA_FINAL."""
    if epoch < BETA_WARMUP_START:
        return 0.0
    if epoch < BETA_WARMUP_END:
        progress = (epoch - BETA_WARMUP_START) / (BETA_WARMUP_END - BETA_WARMUP_START)
        return BETA_FINAL * progress
    return BETA_FINAL


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU.")
    log.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    obs_all, plan_all = load_dataset(args.data_path)
    n_total = obs_all.shape[0]

    # ------------------------------------------------------------------
    # 80/20 train/val split (deterministic)
    # ------------------------------------------------------------------
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    log.info(f"Split: {n_train:,} train / {n_val:,} val")

    generator = torch.Generator().manual_seed(42)
    full_dataset = TensorDataset(obs_all, plan_all)
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=generator,
    )

    # Extract training tensors for stats
    train_indices = train_dataset.indices
    obs_train_raw = obs_all[train_indices]
    plan_train_raw = plan_all[train_indices]

    # ------------------------------------------------------------------
    # Normalization statistics (training split only)
    # ------------------------------------------------------------------
    log.info("Computing normalization statistics on training split ...")
    obs_mean, obs_std, plan_mean, plan_std = compute_normalization_stats(
        obs_train_raw, plan_train_raw,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    ArmPlanCVAE = _load_cvae_class()
    model = ArmPlanCVAE().to(device)
    model.set_normalization_stats(
        obs_mean.to(device),
        obs_std.to(device),
        plan_mean.to(device),
        plan_std.to(device),
    )
    log.info(f"Total parameters:   {model.num_parameters():,}")
    log.info(f"Encoder parameters: {model.num_encoder_parameters():,}")

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )
    recon_loss_fn = nn.MSELoss(reduction="mean")

    # Pre-compute normalized plan targets for reconstruction loss
    plan_std_dev = plan_std.to(device)
    plan_mean_dev = plan_mean.to(device)

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
    # Training loop with early stopping on val ELBO
    # ------------------------------------------------------------------
    best_val_elbo = float("inf")
    patience_counter = 0
    best_state_dict = None

    log.info(
        f"Starting training for up to {args.epochs} epochs "
        f"(early stop patience={args.patience})"
    )
    log.info(
        f"  batch_size={args.batch_size}, "
        f"n_train={n_train:,}, n_val={n_val:,}"
    )
    log.info(
        f"  Beta schedule: 0.0 for epochs 0-{BETA_WARMUP_START-1}, "
        f"ramp to {BETA_FINAL} by epoch {BETA_WARMUP_END-1}, hold"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        beta = get_beta(epoch)

        # ---- Train ----
        model.train()
        train_recon_accum = 0.0
        train_kl_accum = 0.0
        train_n = 0

        for obs_b, plan_b in train_loader:
            obs_b = obs_b.to(device, non_blocking=True)
            plan_b = plan_b.to(device, non_blocking=True)
            batch_n = obs_b.shape[0]

            z, plan_hat, mu, logvar = model.forward_train(obs_b, plan_b)

            # Reconstruction in normalized space
            plan_b_norm = (plan_b - plan_mean_dev) / plan_std_dev.clamp(min=1e-6)
            recon = recon_loss_fn(plan_hat, plan_b_norm)
            kl = kl_divergence(mu, logvar)
            loss = recon + beta * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_recon_accum += recon.item() * batch_n
            train_kl_accum += kl.item() * batch_n
            train_n += batch_n

        scheduler.step()
        avg_train_recon = train_recon_accum / train_n
        avg_train_kl = train_kl_accum / train_n
        avg_train_elbo = avg_train_recon + beta * avg_train_kl

        # ---- Validate ----
        model.eval()
        val_recon_accum = 0.0
        val_kl_accum = 0.0
        val_n = 0

        with torch.no_grad():
            for obs_b, plan_b in val_loader:
                obs_b = obs_b.to(device, non_blocking=True)
                plan_b = plan_b.to(device, non_blocking=True)
                batch_n = obs_b.shape[0]

                z, plan_hat, mu, logvar = model.forward_train(obs_b, plan_b)
                plan_b_norm = (plan_b - plan_mean_dev) / plan_std_dev.clamp(min=1e-6)
                recon = recon_loss_fn(plan_hat, plan_b_norm)
                kl = kl_divergence(mu, logvar)

                val_recon_accum += recon.item() * batch_n
                val_kl_accum += kl.item() * batch_n
                val_n += batch_n

        avg_val_recon = val_recon_accum / val_n
        avg_val_kl = val_kl_accum / val_n
        avg_val_elbo = avg_val_recon + beta * avg_val_kl
        elapsed = time.time() - t0

        log.info(
            f"Epoch {epoch:>4}/{args.epochs}  "
            f"recon={avg_train_recon:.6f}  KL={avg_train_kl:.4f}  "
            f"beta={beta:.3f}  ELBO={avg_train_elbo:.6f}  "
            f"val_recon={avg_val_recon:.6f}  val_KL={avg_val_kl:.4f}  "
            f"val_ELBO={avg_val_elbo:.6f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.1f}s"
        )

        # ---- Early stopping ----
        # During beta warmup, track reconstruction loss only (the KL term
        # is artificially suppressed so ELBO is not a reliable metric).
        # Once beta reaches its final value, switch to full ELBO.
        if beta >= BETA_FINAL:
            metric = avg_val_elbo
            metric_name = "val_ELBO"
        else:
            metric = avg_val_recon
            metric_name = "val_recon"

        if metric < best_val_elbo:
            best_val_elbo = metric
            patience_counter = 0
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            log.info(
                f"  -> New best {metric_name}={best_val_elbo:.6f}. "
                f"Checkpoint cached."
            )
        else:
            patience_counter += 1
            log.info(
                f"  -> No improvement ({patience_counter}/{args.patience})"
            )
            if patience_counter >= args.patience:
                log.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # ------------------------------------------------------------------
    # Final evaluation with best weights
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Final evaluation with best checkpoint weights ...")
    model.load_state_dict(
        {k: v.to(device) for k, v in best_state_dict.items()}
    )
    model.eval()

    final_recon = 0.0
    final_kl = 0.0
    final_n = 0
    with torch.no_grad():
        for obs_b, plan_b in val_loader:
            obs_b = obs_b.to(device, non_blocking=True)
            plan_b = plan_b.to(device, non_blocking=True)
            batch_n = obs_b.shape[0]

            z, plan_hat, mu, logvar = model.forward_train(obs_b, plan_b)
            plan_b_norm = (plan_b - plan_mean_dev) / plan_std_dev.clamp(min=1e-6)
            recon = recon_loss_fn(plan_hat, plan_b_norm)
            kl = kl_divergence(mu, logvar)

            final_recon += recon.item() * batch_n
            final_kl += kl.item() * batch_n
            final_n += batch_n

    final_recon /= final_n
    final_kl /= final_n

    log.info(f"Val reconstruction MSE: {final_recon:.6f}")
    log.info(f"Val KL divergence:      {final_kl:.6f}")
    log.info(f"Val ELBO (beta=1.0):    {final_recon + final_kl:.6f}")

    if final_kl < 1.0:
        log.warning(
            f"KL = {final_kl:.4f} < 1.0 — possible posterior collapse! "
            "The latent may not be encoding meaningful information. "
            "Consider reducing beta or increasing latent_dim."
        )

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = model.cpu()
    model_cpu.load_state_dict({k: v for k, v in best_state_dict.items()})

    torch.save(
        {
            "model_state_dict": model_cpu.state_dict(),
            "train_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": args.device,
                "data_path": args.data_path,
                "patience": args.patience,
                "n_train": n_train,
                "n_val": n_val,
                "beta_warmup_start": BETA_WARMUP_START,
                "beta_warmup_end": BETA_WARMUP_END,
                "beta_final": BETA_FINAL,
                "latent_dim": LATENT_DIM,
            },
            "metrics": {
                "val_recon_mse": final_recon,
                "val_kl": final_kl,
                "val_elbo": final_recon + final_kl,
                "best_val_elbo": best_val_elbo,
            },
            "normalization_stats": {
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "plan_mean": plan_mean,
                "plan_std": plan_std,
            },
        },
        save_path,
    )
    log.info(f"Checkpoint saved to: {save_path}")
    log.info(f"Load with: model.load_frozen('{save_path}')")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CVAE training for the AnticiPose B6 arm plan encoder.",
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
        default="checkpoints/arm_plan_cvae.pt",
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
        help="Early stopping patience (epochs without val ELBO improvement).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device: 'cuda', 'cuda:0', 'cpu', etc.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
