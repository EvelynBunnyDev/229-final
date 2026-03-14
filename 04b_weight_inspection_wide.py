"""
04b_weight_inspection_wide.py
────────────────────────────────────────────────────────────────────
Inspect, visualize and compare the learned weights across all 4
Simple3D VideoModel checkpoints.

  ┌─ BACKBONE (ann.*):
  │   conv1 filters — 32 filters of shape (64, 3, 5, 11, 11)
  │                   what motion patterns does each layer detect?
  │   conv2-7 filters — (64, 64, 3, 3, 3)  growing complexity
  │
  ├─ READOUT (w_s, w_f):
  │   w_s — spatial receptive fields learned per neuron
  │   w_f — feature tuning curves learned per neuron
  │
  └─ CALIBRATION (ann_bn):
       running_mean/var — tells us how active each (channelxtime)
       feature is on average across the training set

Plots produced:
  1. filter_stats.png       — mean|weight|, std, dead% per layer, all models
  2. weight_distributions.png — conv1 weight histograms, all models
  3. ws_receptive_fields.png  — spatial weight map for first 16 neurons (w_s)
  4. wf_tuning_curves.png     — feature tuning curves for first 16 neurons (w_f)
  5. conv1_filters.png        — visualize all 32 conv1 filters (center time slice)
  6. convN_filters.png        — same for each deeper layer

Usage:
    python 04b_weight_inspection_wide.py --ckpt_dir . --out_dir weight_plots/
    python 04b_weight_inspection_wide.py --ckpt_dir . --model simple3d5 --out_dir weight_plots/
"""
# *** mid_channels=16 (wide) variant — 64 output channels per layer ***
# *** Checkpoint suffix: _64_wide.pth  (retrained with mid_channels=16) ***


import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict

MODEL_CONFIG = {
    "simple3d1": {"n_layers":1, "ckpt":"dorsal_stream_simple3d1_64_none.pth"},
    "simple3d3": {"n_layers":3, "ckpt":"dorsal_stream_simple3d3_64_none.pth"},
    "simple3d5": {"n_layers":5, "ckpt":"dorsal_stream_simple3d5_64_none.pth"},
    "simple3d7": {"n_layers":7, "ckpt":"dorsal_stream_simple3d7_64_none.pth"},
}

# Exact shapes from simple3d.py (mid_channels=16, 4x8=32 output channels)
LAYER_SHAPES = {
    "ann.conv1.weight": (64, 3, 5, 11, 11), # large spatio-temporal kernel, RGB input
    "ann.conv2.weight": (64, 64, 3, 3, 3), # 3x3x3 from layer 2 onward
    "ann.conv3.weight": (64, 64, 3, 3, 3),
    "ann.conv4.weight": (64, 64, 3, 3, 3),
    "ann.conv5.weight": (64, 64, 3, 3, 3),
    "ann.conv6.weight": (64, 64, 3, 3, 3),
    "ann.conv7.weight": (64, 64, 3, 3, 3),
}

MODEL_COLORS = {
    "simple3d1": "#ff6b9d",
    "simple3d3": "#c77dff",
    "simple3d5": "#4cc9f0",
    "simple3d7": "#7bf1a8",
}


def load_state(model_name: str, ckpt_dir: str) -> OrderedDict:
    ckpt = os.path.join(ckpt_dir, MODEL_CONFIG[model_name]["ckpt"])
    raw  = torch.load(ckpt, map_location="cpu", weights_only=False)
    # Remap inc_features.conv* → inc_features.model.conv* to match FeatureExtractor
    state = OrderedDict()
    for k, v in raw.items():
        if k.startswith("inc_features.") and not k.startswith("inc_features.model."):
            state[f"inc_features.model.{k[len('inc_features.'):]}"] = v
        else:
            state[k] = v
    return state


def compute_stats(state: OrderedDict) -> dict:
    stats = {}
    for key, t in state.items():
        if "num_batches" in key:
            continue
        a = t.float().numpy().flatten()
        stats[key] = {
            "mean":     float(a.mean()),
            "std":      float(a.std()),
            "abs_mean": float(np.abs(a).mean()),
            "dead_pct": float((np.abs(a) < 1e-6).mean() * 100),
            "n":        len(a),
            "shape":    tuple(t.shape),
        }
    return stats


def print_report(all_stats: dict):
    print(f"\n{'═'*90}")
    print("  WEIGHT INSPECTION REPORT")
    print(f"{'═'*90}")
    for model_name, stats in all_stats.items():
        print(f"\n  ◆ {model_name.upper()}")
        print(f"  {'Layer':<42} {'Shape':<24} {'AbsMean':>8} {'Std':>8} {'Dead%':>7} {'N':>10}")
        print(f"  {'─'*42} {'─'*24} {'─'*8} {'─'*8} {'─'*7} {'─'*10}")
        for k, s in stats.items():
            print(f"  {k:<42} {str(s['shape']):<24} "
                  f"{s['abs_mean']:>8.4f} {s['std']:>8.4f} "
                  f"{s['dead_pct']:>6.1f}% {s['n']:>10,}")


# ─────────────────────────────────────────────────────────────────
#  Plot 1 — Filter statistics per layer across models
# ─────────────────────────────────────────────────────────────────

def plot_filter_stats(all_stats: dict, out_path: str):
    metrics = [
        ("abs_mean", "Mean |Weight|",      "Filter activation strength"),
        ("std",      "Weight Std Dev",     "Filter diversity"),
        ("dead_pct", "Dead Weight %",      "% weights ≈ 0  (inactive filters)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), facecolor="#09090f")
    fig.suptitle("Layer-by-Layer Weight Statistics  —  All Simple3D Models",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    for ax, (metric, ylabel, desc) in zip(axes, metrics):
        ax.set_facecolor("#111120")
        ax.set_title(f"{ylabel}  ·  {desc}", color="#888899", fontsize=10, pad=6)
        ax.set_ylabel(ylabel, color="#777788", fontsize=9)
        ax.tick_params(colors="#444466", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#222233")
        ax.grid(axis="y", color="#1a1a2e", linewidth=0.6)

        for model_name, stats in all_stats.items():
            conv_keys = [k for k in stats
                         if k.startswith("ann.conv") and k.endswith(".weight")]
            vals   = [stats[k][metric] for k in conv_keys]
            labels = [k.replace("ann.","").replace(".weight","") for k in conv_keys]
            color  = MODEL_COLORS.get(model_name, "white")
            ax.plot(range(len(vals)), vals, "o-", color=color,
                    label=model_name, linewidth=2, markersize=7, alpha=0.9)
            if model_name == "simple3d7":
                for xi, (v, lab) in enumerate(zip(vals, labels)):
                    ax.annotate(lab, (xi, v), textcoords="offset points",
                                xytext=(0,9), ha="center", color="#444466", fontsize=7)

        ax.legend(facecolor="#14142a", edgecolor="#222244",
                  labelcolor="white", fontsize=9, loc="best")
        ax.set_xticks([])

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Plot 2 — Weight value distributions (conv1 across models)
# ─────────────────────────────────────────────────────────────────

def plot_weight_distributions(all_states: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#09090f")
    ax.set_facecolor("#111120")
    ax.set_title("conv1 Weight Distributions  —  All Models\n"
                 "(ann.conv1.weight — shape 32x3x5x11x11 — 58,080 values each)",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Weight value", color="#777788")
    ax.set_ylabel("Density",      color="#777788")
    ax.tick_params(colors="#444466")
    for sp in ax.spines.values(): sp.set_edgecolor("#222233")
    ax.grid(axis="y", color="#1a1a2e", linewidth=0.6)

    for name, state in all_states.items():
        key = "ann.conv1.weight"
        if key not in state: continue
        arr = state[key].float().numpy().flatten()
        ax.hist(arr, bins=100, density=True, alpha=0.5,
                color=MODEL_COLORS[name], label=name, linewidth=0)

    ax.legend(facecolor="#14142a", edgecolor="#222244",
              labelcolor="white", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Plot 3 — Spatial receptive fields (w_s)
# ─────────────────────────────────────────────────────────────────

def plot_receptive_fields(state: OrderedDict, model_name: str,
                           out_path: str, n_neurons: int = 16):
    """
    Visualize w_s for the first n_neurons.
    w_s shape: (n_neurons, 1, H*W, 1)  — reshape to (H,W) per neuron.
    |w_s| is used (absolute value — as in the forward pass).
    """
    if "w_s" not in state:
        print("  ⚠  w_s not found in checkpoint — skipping receptive field plot")
        return

    ws   = torch.abs(state["w_s"]).squeeze().numpy()   # (n_neurons, H*W)
    n    = min(n_neurons, ws.shape[0])
    side = int(np.sqrt(ws.shape[1]))                    # infer H = W from H*W
    if side * side != ws.shape[1]:
        print(f"  ⚠  w_s spatial dim {ws.shape[1]} is not square — cannot reshape")
        return

    n_cols = 8
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols*1.8, n_rows*1.9 + 1),
                             facecolor="#09090f")
    fig.suptitle(
        f"Spatial Receptive Fields (|w_s|)  ·  {model_name}\n"
        f"Each panel = where in the {side}x{side} feature map one neuron attends",
        color="white", fontsize=11, fontweight="bold")

    color = MODEL_COLORS.get(model_name, "white")
    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")
        if i < n:
            rf = ws[i].reshape(side, side)
            rf = (rf - rf.min()) / (rf.max() - rf.min() + 1e-8)
            ax.imshow(rf, cmap="inferno", vmin=0, vmax=1)
            ax.set_title(f"n{i}", color="#555577", fontsize=6.5, pad=1.5)

    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Plot 4 — Feature tuning curves (w_f)
# ─────────────────────────────────────────────────────────────────

def plot_feature_tuning(state: OrderedDict, model_name: str,
                         out_path: str, n_neurons: int = 16):
    """
    Visualize w_f for the first n_neurons.
    w_f shape: (1, n_neurons, 1, C*T=320)  — 160 = 64 channels x 5 frames.
    n_neurons = 2244 in these checkpoints.
    Shows which of the 160 timexchannel features each neuron weights.
    """
    if "w_f" not in state:
        print("  ⚠  w_f not found — skipping tuning curve plot")
        return

    wf   = state["w_f"].squeeze().numpy()       # (n_neurons, 160)
    n    = min(n_neurons, wf.shape[0])
    feat = wf.shape[1]                           # 160
    T, C = 5, feat // 5                          # 5 frames, 64 channels

    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols*4.5, n_rows*2.2 + 1),
                             facecolor="#09090f")
    fig.suptitle(
        f"Feature Tuning Curves (w_f)  ·  {model_name}\n"
        f"Each panel = which of the {feat} timexchannel features neuron N responds to\n"
        f"(reshaped to {T} frames x {C} channels — bright = stronger tuning)",
        color="white", fontsize=11, fontweight="bold")

    color = MODEL_COLORS.get(model_name, "white")
    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")
        if i < n:
            curve = wf[i].reshape(T, C)         # (5, 32)
            vmax  = max(abs(curve.max()), abs(curve.min()))
            im    = ax.imshow(curve, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                              aspect="auto")
            ax.set_title(f"neuron {i}", color="#555577", fontsize=7.5, pad=2)
            ax.set_xlabel("channel", color="#333355", fontsize=6)
            ax.set_ylabel("frame",   color="#333355", fontsize=6)

    plt.tight_layout(rect=[0,0,1,0.90])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Plot 5 — Learned conv filter visualization
# ─────────────────────────────────────────────────────────────────

def plot_conv_filters(state: OrderedDict, model_name: str,
                       layer_key: str, out_path: str):
    """
    Visualize learned 3D conv filters.

    conv1: shape (64, 3, 5, 11, 11) — show center time slice, mean across RGB
    conv2+: shape (64, 64, 3, 3, 3) — show center time slice, mean across input channels
    Each subplot = one output filter.
    """
    if layer_key not in state:
        return

    w     = state[layer_key].float().numpy()   # (out, in, kT, kH, kW)
    n_out = w.shape[0]
    t_ctr = w.shape[2] // 2                    # center temporal slice

    # collapse input channels to single image
    if w.shape[1] == 3:
        # RGB → show as grayscale mean
        filt_2d = w[:, :, t_ctr, :, :].mean(axis=1)   # (32, kH, kW)
    else:
        filt_2d = w[:, :, t_ctr, :, :].mean(axis=1)   # (32, kH, kW)

    n_cols = 8
    n_rows = (n_out + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols*1.7, n_rows*1.8 + 1.2),
                             facecolor="#09090f")
    layer_short = layer_key.replace("ann.","").replace(".weight","")
    fig.suptitle(
        f"Learned Filters  ·  {model_name}  ·  {layer_short}  "
        f"(center time slice t={t_ctr})\n"
        f"Shape: {tuple(w.shape)}  ·  Red=positive, Blue=negative",
        color="white", fontsize=10, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#0c0c18")
        ax.axis("off")
        if i < n_out:
            filt = filt_2d[i]
            vmax = max(abs(filt.max()), abs(filt.min()))
            ax.imshow(filt, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"f{i}", color="#444466", fontsize=6, pad=1)

    plt.tight_layout(rect=[0,0,1,0.90])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weight inspection for Simple3D VideoModels")
    parser.add_argument("--ckpt_dir", default=".",             help="Dir with .pth files")
    parser.add_argument("--out_dir",  default="weight_plots",  help="Output dir for plots")
    parser.add_argument("--model",    default="all",
                        help="Model to run, or 'all'")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = list(MODEL_CONFIG.keys()) if args.model == "all" else [args.model]

    # ── Load all requested checkpoints ──
    all_states = {}
    all_stats  = {}
    for name in targets:
        print(f"  Loading {name}…")
        state = load_state(name, args.ckpt_dir)
        all_states[name] = state
        all_stats[name]  = compute_stats(state)

    # ── Print report ──
    print_report(all_stats)

    # ── Cross-model comparison plots (only if >1 model) ──
    if len(all_states) > 1:
        plot_filter_stats(all_stats,
            os.path.join(args.out_dir, "filter_stats.png"))
        plot_weight_distributions(all_states,
            os.path.join(args.out_dir, "weight_distributions.png"))

    # ── Per-model plots ──
    for name, state in all_states.items():
        cfg    = MODEL_CONFIG[name]
        n_layers = cfg["n_layers"]
        print(f"\n  Plotting {name}…")

        # receptive fields
        plot_receptive_fields(state, name,
            os.path.join(args.out_dir, f"{name}_receptive_fields_ws.png"))

        # feature tuning
        plot_feature_tuning(state, name,
            os.path.join(args.out_dir, f"{name}_feature_tuning_wf.png"))

        # filter visualizations for each conv layer
        for i in range(1, n_layers+1):
            key = f"ann.conv{i}.weight"
            plot_conv_filters(state, name, key,
                os.path.join(args.out_dir, f"{name}_conv{i}_filters.png"))

    print(f"\n  Done. All outputs in '{args.out_dir}/'")
