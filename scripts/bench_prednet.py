"""Benchmark the training of a model (PredNet by default, any model config) on the selected device: time per step,
data wait, memory, and a short training run.

The loss is the unweighted one (digit weight 1): PredNet's own loss for PredNet, half the L1 error for other models.
Results go to runs/bench/<model>_b<batch>_t<frames>_<device>/: a copy of the settings, results.json, the loss
curve, predictions on validation sequences, and the trained model (model.pt, for scripts/show_predictions.py). After a
few hundred steps the predictions are still rough: they show that the whole pipeline works, not how well the model does.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt
import numpy as np
import torch

from peekaboo.config import load_config, save_config
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset, make_loader, prepare_batch
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.render import RenderSettings
from peekaboo.data.store import StoredDataset
from peekaboo.device import describe_device, get_device, synchronize
from peekaboo.models import build_model
from peekaboo.models.prednet import count_parameters
from peekaboo.paths import CONFIGS_DIR, DATASETS_DIR, RUNS_DIR
from peekaboo.seeding import seed_everything
from peekaboo.train.checkpoint import save_checkpoint
from peekaboo.train.losses import training_loss


def memory_bytes(device: torch.device) -> int | None:
    """Memory currently held by the device allocator, or None on the CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    return None


def save_predictions(model: torch.nn.Module, batch: dict[str, torch.Tensor], seq_len: int, path: Path) -> dict[str, float]:
    """Draw actual and predicted frames of a few validation sequences, and compare with copying the last frame."""
    model.eval()
    with torch.no_grad():
        frames = batch["frames"][:, :seq_len]
        prediction = model(frames)["prediction"]
    frames, prediction = frames.cpu().numpy(), prediction.cpu().numpy()
    model.train()
    # mean squared error per pixel on frames 1 and later, against copying the previous frame
    model_mse = float(((prediction[:, 1:] - frames[:, 1:]) ** 2).mean())
    copy_mse = float(((frames[:, :-1] - frames[:, 1:]) ** 2).mean())

    shown = list(range(1, seq_len, 3))
    n = frames.shape[0]
    fig, axes = plt.subplots(2 * n, len(shown), figsize=(1.3 * len(shown), 1.0 * 2 * n), squeeze=False)
    for i in range(n):
        for j, t in enumerate(shown):
            for row, (images, label) in enumerate(((frames, "actual"), (prediction, "predicted"))):
                ax = axes[2 * i + row, j]
                ax.imshow(np.clip(images[i, t].transpose(1, 2, 0), 0, 1), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(label, fontsize=6)
                if i == 0 and row == 0:
                    ax.set_title(f"t={t}", fontsize=6)
    fig.suptitle(f"Validation sequences: actual frame t and the model's prediction of it from frames before t "
                 f"(MSE {model_mse:.4f}, copy last frame {copy_mse:.4f})", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"val_mse": model_mse, "val_copy_last_mse": copy_mse}


def main() -> int:
    """Train for a few hundred steps, time every step, and save the results next to a copy of the settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CONFIGS_DIR / "models" / "prednet_5l.yaml")
    parser.add_argument("--data-config", type=Path, default=CONFIGS_DIR / "data" / "base.yaml")
    parser.add_argument("--steps", type=int, default=200, help="timed steps, after the warmup")
    parser.add_argument("--warmup", type=int, default=10, help="steps left out of the timing")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=40, help="frames per sequence (at most the generated 40)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "bench")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    model_config = load_config(args.model)
    data_config = load_config(args.data_config)
    settings = GeneratorSettings.from_config(data_config)
    render_settings = RenderSettings.from_config(data_config)
    digits = data_config["digits"]
    pool = build_digit_pool("train", digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)
    dataset = OnTheFlyDataset(pool, settings, render_settings, "bench_train", args.seed)
    total_steps = args.warmup + args.steps
    loader = make_loader(dataset, args.batch_size, start=0, count=args.batch_size * total_steps,
                         num_workers=args.workers)

    model = build_model(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run_dir = args.out / f"{args.model.stem}_b{args.batch_size}_t{args.seq_len}_{device.type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"model": model_config, "data": data_config, "args": {k: str(v) for k, v in vars(args).items()}}
    save_config(run_config, run_dir / "config.yaml")
    print(f"{args.model.stem}: {count_parameters(model):,} parameters on {describe_device(device)}, "
          f"batch {args.batch_size} x {args.seq_len} frames, {args.workers} workers")

    losses, compute_times, data_times, peak_memory = [], [], [], 0
    iterator = iter(loader)
    for step in range(total_steps):
        start = time.perf_counter()
        batch = prepare_batch(next(iterator), device)
        synchronize(device)
        data_time = time.perf_counter() - start

        start = time.perf_counter()
        frames = batch["frames"][:, :args.seq_len]
        loss = training_loss(model, model(frames), frames)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        synchronize(device)
        compute_time = time.perf_counter() - start

        losses.append(float(loss.item()))
        if step >= args.warmup:
            compute_times.append(compute_time)
            data_times.append(data_time)
        peak_memory = max(peak_memory, memory_bytes(device) or 0)
        if step == 0 or (step + 1) % 50 == 0:
            print(f"step {step + 1:>4}/{total_steps}: loss {losses[-1]:.5f}, compute {compute_time:.3f} s, "
                  f"data wait {data_time:.3f} s")

    save_checkpoint(run_dir / "model.pt", model, optimizer, total_steps, run_config)
    per_step = float(np.mean(compute_times) + np.mean(data_times))
    results = {
        "model": args.model.stem, "parameters": count_parameters(model), "device": describe_device(device),
        "batch_size": args.batch_size, "seq_len": args.seq_len, "workers": args.workers,
        "compute_s_per_step": float(np.mean(compute_times)), "data_wait_s_per_step": float(np.mean(data_times)),
        "s_per_step": per_step, "sequences_per_s": args.batch_size / per_step,
        "peak_memory_gb": peak_memory / 1e9 if peak_memory else None,
        "hours_for_steps": {str(n): n * per_step / 3600 for n in (10_000, 30_000, 50_000)},
        "loss_first": losses[0], "loss_last_10": float(np.mean(losses[-10:])),
    }

    # a figure of the loss, and predictions on validation sequences with an occlusion
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.arange(1, total_steps + 1), losses, lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss (digit weight 1)")
    ax.set_yscale("log")
    ax.set_title(f"{args.model.stem}, batch {args.batch_size}, {args.seq_len} frames", fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "loss.png", dpi=110)
    plt.close(fig)
    val_dir = DATASETS_DIR / "val_v1"
    if val_dir.exists():
        val = StoredDataset(val_dir)
        picks = [i for i, s in enumerate(val.specs) if s.condition == "occlusion"][:3]
        batch = torch.utils.data.default_collate([val[i] for i in picks])
        results.update(save_predictions(model, prepare_batch(batch, device), args.seq_len,
                                        run_dir / "predictions.png"))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\ncompute {results['compute_s_per_step']:.3f} s/step, data wait {results['data_wait_s_per_step']:.3f} "
          f"s/step, {results['sequences_per_s']:.1f} sequences/s")
    if results["peak_memory_gb"] is not None:
        print(f"peak memory {results['peak_memory_gb']:.2f} GB")
    print("estimated training time: " + ", ".join(f"{int(n):,} steps {h:.1f} h"
                                                  for n, h in results["hours_for_steps"].items()))
    print(f"loss {results['loss_first']:.5f} at the first step, {results['loss_last_10']:.5f} over the last 10")
    if "val_mse" in results:
        print(f"validation MSE {results['val_mse']:.5f} (copy last frame {results['val_copy_last_mse']:.5f}); "
              f"after so few steps, the model is not expected to beat the copy yet")
    print(f"saved in {run_dir}")
    print(f"draw the predictions with: python -m scripts.show_predictions --run {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
