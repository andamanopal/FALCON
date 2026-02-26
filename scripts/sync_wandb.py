"""Re-log TensorBoard events to WandB for a completed training run.

WandB's sync_tensorboard=True often stops mid-run (TBDirWatcher dies silently).
This reads the TB events file directly and logs all scalars to a fresh WandB run.

Usage:
    python scripts/sync_wandb.py <run_dir> <run_name>
    python scripts/sync_wandb.py logs/anticipose_overnight/20260226_033841-B1_reactive_seed35-... B1_reactive_seed35
"""
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import wandb

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <run_dir> <run_name>")
        sys.exit(1)

    run_dir = sys.argv[1]
    run_name = sys.argv[2]

    ea = EventAccumulator(run_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"[wandb] No scalar tags found in {run_dir}")
        return

    step_data = {}
    for tag in tags:
        for event in ea.Scalars(tag):
            if event.step not in step_data:
                step_data[event.step] = {}
            step_data[event.step][tag] = event.value

    print(f"[wandb] Found {len(tags)} tags, {len(step_data)} steps from {run_dir}")
    wandb.init(
        project="AnticiPose",
        entity="andaman-l",
        name=run_name,
        tags=["resync"],
    )
    for step in sorted(step_data.keys()):
        wandb.log(step_data[step], step=step)
    wandb.finish()
    print("[wandb] Sync complete")

if __name__ == "__main__":
    main()
