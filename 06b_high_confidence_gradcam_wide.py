"""
06b_high_confidence_gradcam_wide.py
────────────────────────────────────────────────────────────────────
Find videos and neurons where BOTH conditions are true:
  1. Actual recorded firing rate is high (the neuron really responded)
  2. Predicted firing rate is high (the model got it right)

These are the "ground truth confirmed" cases — the model's explanation
is biologically meaningful because we know the neuron genuinely fired.

Then run Grad-CAM on those neurons for that video and produce:
  - A population heatmap overlay (all top-N neurons on one stimulus)
  - Individual Grad-CAM panels per neuron with actual vs predicted rate

WHAT THE PICKLE CONTAINS (from dataloader.py):
  data['train_stimuli']   — list of video filenames (no extension)
  data['train_activity']  — array (n_videos, n_neurons) actual firing rates
  data['test_stimuli']    — same for test set
  data['test_activity']   — same for test set

Usage:
    python 06b_high_confidence_gradcam_wide.py \\
        --dataset_dir ./dataset \\
        --ckpt_dir    . \\
        --out_dir     high_confidence_gradcam/ \\
        --top_n       20

    # Use test set (default) or train set:
    python 06b_high_confidence_gradcam_wide.py --split test --top_n 20
"""
# *** mid_channels=16 (wide) variant — 64 output channels per layer ***
# *** Checkpoint suffix: _64_none.pth  (retrained with mid_channels=16) ***


import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import v2


# ─────────────────────────────────────────────────────────────────
#  Backbone classes (exact copy of simple3d.py)
# ─────────────────────────────────────────────────────────────────

class Simple3DConvNet1(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16):
        super().__init__()
        ch = 4 * mid_channels
        self.conv1   = nn.Conv3d(in_channels, ch, kernel_size=(5,11,11), stride=(1,2,2), padding=(2,5,5))
        self.bn1     = nn.BatchNorm3d(ch)
        self.avgpool = nn.AvgPool3d(kernel_size=(1,4,4), stride=(1,4,4))
    def forward(self, x):
        return self.avgpool(F.relu(self.bn1(self.conv1(x))))

class Simple3DConvNet3(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16):
        super().__init__()
        ch = 4 * mid_channels
        self.conv1 = nn.Conv3d(in_channels, ch, kernel_size=(5,11,11), stride=(1,2,2), padding=(2,5,5)); self.bn1 = nn.BatchNorm3d(ch)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=3, stride=(1,2,2), padding=1);                        self.bn2 = nn.BatchNorm3d(ch)
        self.conv3 = nn.Conv3d(ch, ch, kernel_size=3, stride=(1,2,2), padding=1);                        self.bn3 = nn.BatchNorm3d(ch)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); return F.relu(self.bn3(self.conv3(x)))

class Simple3DConvNet5(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16):
        super().__init__()
        ch = 4 * mid_channels
        self.conv1 = nn.Conv3d(in_channels, ch, kernel_size=(5,11,11), stride=(1,2,2), padding=(2,5,5)); self.bn1 = nn.BatchNorm3d(ch)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=3, stride=(1,2,2), padding=1);                        self.bn2 = nn.BatchNorm3d(ch)
        self.conv3 = nn.Conv3d(ch, ch, kernel_size=3, stride=(1,2,2), padding=1);                        self.bn3 = nn.BatchNorm3d(ch)
        self.conv4 = nn.Conv3d(ch, ch, kernel_size=3, stride=(1,1,1), padding=1);                        self.bn4 = nn.BatchNorm3d(ch)
        self.conv5 = nn.Conv3d(ch, ch, kernel_size=3, stride=(1,1,1), padding=1);                        self.bn5 = nn.BatchNorm3d(ch)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x))); return F.relu(self.bn5(self.conv5(x)))

class Simple3DConvNet7(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16):
        super().__init__()
        ch = 4 * mid_channels
        self.conv1 = nn.Conv3d(in_channels, ch, kernel_size=(5,11,11), stride=(1,2,2), padding=(2,5,5)); self.bn1 = nn.BatchNorm3d(ch)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=3, stride=(1,2,2), padding=1);                        self.bn2 = nn.BatchNorm3d(ch)
        self.conv3 = nn.Conv3d(ch, ch, kernel_size=3, stride=(1,2,2), padding=1);                        self.bn3 = nn.BatchNorm3d(ch)
        self.conv4 = nn.Conv3d(ch, ch, kernel_size=3, stride=(1,1,1), padding=1);                        self.bn4 = nn.BatchNorm3d(ch)
        self.conv5 = nn.Conv3d(ch, ch, kernel_size=3, stride=(1,1,1), padding=1);                        self.bn5 = nn.BatchNorm3d(ch)
        self.conv6 = nn.Conv3d(ch, ch, kernel_size=3, stride=(1,1,1), padding=1);                        self.bn6 = nn.BatchNorm3d(ch)
        self.conv7 = nn.Conv3d(ch, ch, kernel_size=3, stride=(1,1,1), padding=1);                        self.bn7 = nn.BatchNorm3d(ch)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x))); x = F.relu(self.bn5(self.conv5(x))); x = F.relu(self.bn6(self.conv6(x)))
        return F.relu(self.bn7(self.conv7(x)))

BACKBONE_CLASSES = {"simple3d1": Simple3DConvNet1, "simple3d3": Simple3DConvNet3,
                    "simple3d5": Simple3DConvNet5, "simple3d7": Simple3DConvNet7}

MODEL_CONFIG = {
    "simple3d1": {"tap": "conv1", "ckpt": "dorsal_stream_simple3d1_64_none.pth"},
    "simple3d3": {"tap": "conv3", "ckpt": "dorsal_stream_simple3d3_64_none.pth"},
    "simple3d5": {"tap": "conv5", "ckpt": "dorsal_stream_simple3d5_64_none.pth"},
    "simple3d7": {"tap": "conv7", "ckpt": "dorsal_stream_simple3d7_64_none.pth"},
}


# ─────────────────────────────────────────────────────────────────
#  VideoModel for Grad-CAM
# ─────────────────────────────────────────────────────────────────

class VideoModelForGradCAM(nn.Module):
    def __init__(self, backbone, tap_layer, n_neurons,
                 input_shape=(1,3,5,64,64), device="cpu"):
        super().__init__()
        self.tap_layer   = tap_layer
        self.ann         = backbone
        self._tap_output = None

        target = dict([*self.ann.named_modules()])[tap_layer]
        target.register_forward_hook(self._tap_hook)

        with torch.no_grad():
            self.ann(torch.ones(input_shape))

        fs = self._tap_output
        B, C, T, H, W = fs.shape
        self.C, self.T, self.H, self.W = C, T, H, W
        feat_dim = C * T

        self.ann_bn = nn.BatchNorm2d(feat_dim, affine=False)
        self.w_s    = nn.Parameter(torch.ones(n_neurons, 1, H*W, 1))
        self.w_f    = nn.Parameter(torch.ones(1, n_neurons, 1, feat_dim))
        nn.init.xavier_normal_(self.w_s)
        nn.init.xavier_normal_(self.w_f)

    def _tap_hook(self, module, inp, output):
        self._tap_output = output
        if output.requires_grad:
            output.retain_grad()

    def forward(self, x):
        self.ann(x)
        f = self._tap_output
        f = f.view(f.shape[0], f.shape[1]*f.shape[2], f.shape[3], f.shape[4])
        f = self.ann_bn(f)
        f = f.view(f.shape[0], f.shape[1], f.shape[2]*f.shape[3], 1)
        f = f.permute(0, -1, 2, 1)
        f = F.conv2d(f, torch.abs(self.w_s))
        f = torch.mul(f, self.w_f)
        f = torch.sum(f, -1, keepdim=True)
        return f.squeeze(-1) # (B, n_neurons, 1)


def load_model(model_name, ckpt_dir, n_neurons=2244, stimulus_size=64, device="cpu"):
    cfg      = MODEL_CONFIG[model_name]
    ckpt     = os.path.join(ckpt_dir, cfg["ckpt"])
    backbone = BACKBONE_CLASSES[model_name]()
    model    = VideoModelForGradCAM(backbone, cfg["tap"], n_neurons,
                                    (1,3,5,stimulus_size,stimulus_size), device)
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    state = {}
    for k, v in raw.items():
        if k.startswith("inc_features.") and not k.startswith("inc_features.model."):
            state[f"inc_features.model.{k[len('inc_features.'):]}"] = v
        else:
            state[k] = v
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def compute_gradcam(model, video_tensor, neuron_idx):
    """Returns (T, H, W) normalized CAM for one neuron."""
    model.zero_grad()
    model.ann.train()
    out   = model(video_tensor)
    score = out[0, neuron_idx, 0]
    tap   = model._tap_output
    grads = torch.autograd.grad(score, tap,
                                retain_graph=False,
                                create_graph=False)[0]
    model.ann.eval()

    alphas = grads[0].mean(dim=[1, 2, 3]) # (C,)
    acts   = tap[0].detach() # (C, T, H, W)
    cam    = torch.zeros(acts.shape[1:])
    for c in range(acts.shape[0]):
        cam = cam + alphas[c] * acts[c]
    cam = F.relu(cam)
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam.detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────

def center_crop(img):
    return v2.functional.crop(img, top=115, left=260, height=150, width=150)

def get_transform(stimulus_size=64):
    return T.Compose([
        T.Lambda(center_crop),
        T.Resize(stimulus_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def load_pickle(dataset_dir, name="dorsal_stream"):
    path = os.path.join(dataset_dir, f"{name}_dataset.pickle")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_video_tensor(video_path, stimulus_size=64, num_frames=5):
    """Load one .mp4 → (1, 3, 5, H, W) tensor."""
    transform = get_transform(stimulus_size)
    cap    = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(transform(image))
    cap.release()
    if len(frames) < num_frames:
        raise ValueError(f"Video {video_path} has fewer than {num_frames} frames")
    video = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0)  # (1,3,5,H,W)
    return video

def load_video_rgb_for_display(video_path, stimulus_size=64, num_frames=5):
    """Load one .mp4 → (3, 5, H, W) unnormalized float for display."""
    cap    = cv2.VideoCapture(video_path)
    frames = []
    resize = T.Compose([T.Lambda(center_crop), T.Resize(stimulus_size), T.ToTensor()])
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(resize(image))
    cap.release()
    return torch.stack(frames).permute(1, 0, 2, 3).numpy() # (3, 5, H, W)


# ─────────────────────────────────────────────────────────────────
#  find best (video, neuron) pairs
# ─────────────────────────────────────────────────────────────────

def find_high_confidence_neurons(model, data, stimuli, activity,
                                 dataset_dir, stimulus_size,
                                 top_neurons=20,
                                 min_actual_percentile=25,
                                 device="cpu"):
    """
    NEURON SELECTION: ratio closest to 1.0
    ─────────────────────────────────────────
    For each video, selects top_neurons by ranking on:

        |log(predicted / actual)|

    This equals 0 when the model is perfectly calibrated (pred == actual),
    and grows symmetrically whether the model over- or under-predicts.

    GUARD: neurons where actual firing < p{min_actual_percentile} are
    excluded -- ratio is meaningless when a neuron barely fired
    (e.g. actual=0.001 inflates any prediction into a huge ratio).

    ALL valid videos are returned, sorted best-to-worst by their top
    neuron's |log(ratio)|. Rank 1 = most perfectly calibrated video.
    """
    video_dir = os.path.join(dataset_dir, "dorsal_stream")
    n_videos  = len(stimuli)

    # Per-neuron floor: actual rate must exceed this to be included
    actual_floor = np.percentile(activity, min_actual_percentile, axis=0)  # (n_neurons,)

    print(f"\n  Scanning {n_videos} videos x {activity.shape[1]} neurons...")
    print(f"  Selection: top {top_neurons} neurons by |log(pred/actual)| closest to 0")
    print(f"  Guard    : actual rate must exceed p{min_actual_percentile} per neuron")

    all_results = []

    for v_idx, stim_name in enumerate(stimuli):
        video_path = os.path.join(video_dir, stim_name + ".mp4")
        if not os.path.exists(video_path):
            continue

        try:
            video_tensor = load_video_tensor(video_path, stimulus_size).to(device)
        except Exception as e:
            print(f"    Warning  Skipping {stim_name}: {e}")
            continue

        with torch.no_grad():
            pred = model(video_tensor).squeeze(-1).squeeze(0).cpu().numpy()  # (n_neurons,)

        actual = activity[v_idx]                                # (n_neurons,)

        # Only neurons that fired meaningfully AND have positive predictions
        valid_mask = (actual > actual_floor) & (actual > 0) & (pred > 0)
        if valid_mask.sum() == 0:
            continue

        ratio     = np.where(valid_mask, pred / (actual + 1e-8), np.inf)
        log_ratio = np.where(valid_mask, np.abs(np.log(ratio + 1e-8)), np.inf)

        # Ascending sort: 0 = perfect ratio, larger = further from 1.0
        sorted_idxs     = np.argsort(log_ratio)
        top_neuron_idxs = np.array([i for i in sorted_idxs if valid_mask[i]])[:top_neurons]

        if len(top_neuron_idxs) == 0:
            continue

        all_results.append({
            "video_idx":      v_idx,
            "stim_name":      stim_name,
            "video_path":     video_path,
            "video_tensor":   video_tensor,
            "pred_raw":       pred,
            "actual_raw":     actual,
            "ratio":          ratio,
            "log_ratio":      log_ratio,
            "top_neurons":    top_neuron_idxs,
            "best_log_ratio": float(log_ratio[top_neuron_idxs[0]]),
            "best_ratio":     float(ratio[top_neuron_idxs[0]]),
        })

        if (v_idx + 1) % 20 == 0:
            best_so_far = min(r["best_log_ratio"] for r in all_results)
            print(f"    [{v_idx+1}/{n_videos}]  best |log(ratio)| so far: {best_so_far:.4f}")

    # Sort ALL videos best-to-worst; rank in filename = position in this list
    all_results.sort(key=lambda x: x["best_log_ratio"])

    print(f"\n  Found {len(all_results)} valid videos")
    if all_results:
        b = all_results[0]
        print(f"  Rank 1 video : {b['stim_name']}  "
              f"ratio={b['best_ratio']:.4f}  |log(ratio)|={b['best_log_ratio']:.4f}")

    return all_results


def plot_population_heatmap(video_rgb, cams, neuron_idxs,
                             pred_rates, actual_rates, ratios, log_ratios,
                             stim_name, model_name, rank, out_path):
    """
    Population heatmap across all 5 time steps.

    Layout  (rows x cols):
      Row 0         : raw video frames t=0..4  (grayscale)
      Row 1         : population heatmap per time step
                      weighted by calibration quality (1/|log_ratio|)
      Row 2         : overlay of row0 + row1
      Bottom strip  : temporal importance bar
                      (mean spatial activation per frame, population-averaged)
      Right column  : pred vs actual scatter, colored by |log(ratio)|
    """
    from scipy.ndimage import zoom

    T      = video_rgb.shape[1]                  # 5
    n_nrn  = len(cams)

    # Calibration weights for each neuron (best-calibrated = highest weight)
    cal_w  = 1.0 / (log_ratios + 1e-4)
    cal_w  = cal_w / (cal_w.sum() + 1e-8)        # (n_neurons,)

    # Population CAM: (T, H_cam, W_cam) -- weighted mean across neurons per timestep
    cam_shape = cams[0].shape                    # (T, H_cam, W_cam)
    pop_cam   = np.zeros(cam_shape)
    for i, cam in enumerate(cams):
        pop_cam += cal_w[i] * cam
    if pop_cam.max() > 0:
        pop_cam = (pop_cam - pop_cam.min()) / (pop_cam.max() - pop_cam.min())

    # Per-frame raw video (grayscale, H x W)
    frames = []
    for t in range(T):
        f = video_rgb[:, t].mean(axis=0)         # mean over RGB channels
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        frames.append(f)

    H_frame, W_frame = frames[0].shape

    # Upsample each time slice of pop_cam to match frame resolution
    pop_cam_up = []
    for t in range(T):
        zf = [H_frame / pop_cam.shape[1], W_frame / pop_cam.shape[2]]
        pop_cam_up.append(zoom(pop_cam[t], zf, order=1))

    # ── Figure layout ──
    # 5 video columns + 1 scatter column
    # 3 image rows + 1 bar row
    n_cols_img = T 
    fig = plt.figure(figsize=(n_cols_img * 2.6 + 3.5, 10), facecolor="#09090f")
    fig.suptitle(
        f"Population Grad-CAM  |  Rank #{rank}  |  {model_name}  |  {stim_name}\n"
        f"Top {n_nrn} neurons by pred/actual ratio closest to 1.0  "
        f"(calibration-weighted population heatmap per time step)",
        color="white", fontsize=11, fontweight="bold", y=0.99)

    # GridSpec: 3 image rows + 1 bar row, T+1 cols (last col = scatter)
    gs = gridspec.GridSpec(
        4, T + 1,
        figure=fig,
        height_ratios=[3, 3, 3, 1],
        width_ratios=[3] * T + [3.5],
        hspace=0.04, wspace=0.04,
        left=0.03, right=0.97, top=0.92, bottom=0.04,
    )

    row_labels = ["Input frame", "Pop. CAM", "Overlay"]

    for t in range(T):
        frame = frames[t]
        cam_up_t = pop_cam_up[t]

        for row, (data, cmap, use_overlay) in enumerate([
            (frame,    "gray",    False),
            (cam_up_t, "inferno", False),
            (frame,    "gray",    True),
        ]):
            ax = fig.add_subplot(gs[row, t])
            ax.set_facecolor("#0d0d1a")
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            if use_overlay:
                ax.imshow(cam_up_t, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
            if t == 0:
                ax.set_ylabel(row_labels[row], color="#666688", fontsize=8, labelpad=3)
            if row == 0:
                ax.set_title(f"t={t}", color="#555577", fontsize=9, pad=2)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for sp in ax.spines.values(): sp.set_visible(False)

    # Temporal importance bar (row 3, cols 0..T-1)
    temporal = pop_cam.mean(axis=(1, 2)) # mean spatial per frame (T,)
    ax_bar = fig.add_subplot(gs[3, :T])
    ax_bar.set_facecolor("#111122")
    bar_colors = plt.cm.inferno(temporal / (temporal.max() + 1e-8))
    ax_bar.bar(range(T), temporal, color=bar_colors, width=0.7)
    ax_bar.set_xlim(-0.5, T - 0.5)
    ax_bar.set_xticks(range(T))
    ax_bar.set_xticklabels([f"t={i}" for i in range(T)], color="#555577", fontsize=8)
    ax_bar.set_yticks([])
    ax_bar.set_title("Mean spatial activation per frame  (temporal importance, population)",
                     color="#555577", fontsize=8, pad=2)
    for sp in ax_bar.spines.values(): sp.set_visible(False)

    # Scatter: pred vs actual (rows 0-2, last col)
    ax_sc = fig.add_subplot(gs[:3, T])
    ax_sc.set_facecolor("#111120")
    sc = ax_sc.scatter(actual_rates, pred_rates, c=log_ratios,
                       cmap="RdYlGn_r", s=55, alpha=0.85, edgecolors="none",
                       vmin=0, vmax=max(float(log_ratios.max()), 0.1))
    cb = plt.colorbar(sc, ax=ax_sc, shrink=0.7, pad=0.02)
    cb.set_label("|log(pred/actual)|", color="#777788", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="#444466", labelsize=7)
    lim = max(float(actual_rates.max()), float(pred_rates.max())) * 1.12
    ax_sc.plot([0, lim], [0, lim], "--", color="#555577", lw=1, label="ratio=1")
    ax_sc.set_xlim(0, lim); ax_sc.set_ylim(0, lim)
    ax_sc.set_xlabel("Actual firing rate",    color="#777788", fontsize=8)
    ax_sc.set_ylabel("Predicted firing rate", color="#777788", fontsize=8)
    ax_sc.set_title("Pred vs Actual\n(green = ratio near 1.0)", color="#888899", fontsize=8)
    ax_sc.tick_params(colors="#444466", labelsize=7)
    ax_sc.legend(fontsize=7, facecolor="#14142a", labelcolor="white")
    for sp in ax_sc.spines.values(): sp.set_edgecolor("#222233")
    ax_sc.grid(color="#1a1a2e", lw=0.5)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved  {out_path}")


def plot_individual_neurons(video_rgb, cams, neuron_idxs,
                             pred_rates, actual_rates, ratios, log_ratios,
                             stim_name, model_name, rank, out_path, n_show=20):
    """
    Individual neuron panels across all 5 time steps.

    For each neuron shown, a row of 5 subplots — one per time step.
    Each subplot = video frame overlaid with that neuron's Grad-CAM slice.
    Left label = neuron index + calibration stats.
    Color-coded title: green (ratio near 1.0) → yellow → red (far from 1.0).

    Rows = neurons (up to n_show)
    Cols = time steps t=0..4
    """
    from scipy.ndimage import zoom

    n_show  = min(n_show, len(neuron_idxs))
    T = video_rgb.shape[1]                

    # Per-frame raw video (grayscale)
    frames = []
    for t in range(T):
        f = video_rgb[:, t].mean(axis=0)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        frames.append(f)
    H_frame, W_frame = frames[0].shape

    fig, axes = plt.subplots(
        n_show, T,
        figsize=(T * 2.4, n_show * 2.6 + 1.5),
        facecolor="#09090f",
        squeeze=False,
    )
    fig.suptitle(
        f"Individual Grad-CAM  |  Rank #{rank}  |  {model_name}  |  {stim_name}\n"
        f"Each row = one neuron  |  Each column = one time step  |  "
        f"Ordered best → worst ratio (top row = most calibrated)",
        color="white", fontsize=10, fontweight="bold")

    # Column headers
    for t in range(T):
        axes[0, t].set_title(f"t={t}", color="#555577", fontsize=9, pad=3)

    for i in range(n_show):
        cam = cams[i] # (T, H_cam, W_cam)
        r = ratios[i]
        lr = log_ratios[i]
        nidx = neuron_idxs[i]
        title_color = "#88ff88" if lr < 0.2 else ("#ffcc44" if lr < 0.5 else "#ff6666")

        # Normalize this neuron's CAM globally across all time steps
        # so intensity is comparable across frames
        cam_norm = cam.copy()
        if cam_norm.max() > 0:
            cam_norm = (cam_norm - cam_norm.min()) / (cam_norm.max() - cam_norm.min())

        for t in range(T):
            ax = axes[i, t]
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for sp in ax.spines.values(): sp.set_visible(False)

            # Upsample CAM slice to frame resolution
            cam_t = cam_norm[t]
            zf    = [H_frame / cam_t.shape[0], W_frame / cam_t.shape[1]]
            cam_up = zoom(cam_t, zf, order=1)

            ax.imshow(frames[t], cmap="gray", vmin=0, vmax=1)
            ax.imshow(cam_up, cmap="inferno", alpha=0.55, vmin=0, vmax=1)

            # Left-side label only on first column
            if t == 0:
                ax.set_ylabel(
                    f"n#{nidx}\npred={pred_rates[i]:.2f}\nact={actual_rates[i]:.2f}\n"
                    f"r={r:.2f} |l|={lr:.2f}",
                    color=title_color, fontsize=6.5, labelpad=4,
                    rotation=0, ha="right", va="center",
                )

    plt.tight_layout(rect=[0.08, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved  {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grad-CAM over all 5 time steps — neurons ranked by pred/actual ratio closest to 1.0")
    parser.add_argument("--model",         default="simple3d5",
                        choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--ckpt_dir",      default=".",
                        help="Directory with .pth files")
    parser.add_argument("--dataset_dir",   default="./dataset",
                        help="Directory with dorsal_stream_dataset.pickle and videos")
    parser.add_argument("--split",         default="test",  choices=["train", "test"])
    parser.add_argument("--top_neurons",   default=20,  type=int,
                        help="Neurons per video selected by ratio closest to 1.0")
    parser.add_argument("--n_individual",  default=20,  type=int,
                        help="Neuron rows in individual figure (0 = all top_neurons)")
    parser.add_argument("--min_actual_percentile", default=25, type=int,
                        help="Exclude neurons where actual firing < this percentile")
    parser.add_argument("--stimulus_size", default=64,  type=int)
    parser.add_argument("--out_dir",       default="high_confidence_gradcam")
    parser.add_argument("--device",        default="cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 1: load pickle ──
    print("\n  Loading dataset pickle...")
    data     = load_pickle(args.dataset_dir)
    print("  Pickle keys:", list(data.keys()))
    stimuli  = data[f"{args.split}_stimuli"]
    activity = np.array(data[f"{args.split}_activity"])
    print(f"  {args.split} stimuli : {len(stimuli)} videos")
    print(f"  activity shape: {activity.shape}  "
          f"min={activity.min():.3f}  max={activity.max():.3f}  mean={activity.mean():.3f}")

    # ── Step 2: load model ──
    print(f"\n  Loading {args.model}...")
    model = load_model(args.model, args.ckpt_dir,
                       n_neurons=activity.shape[1],
                       stimulus_size=args.stimulus_size,
                       device=args.device)

    # ── Step 3: rank ALL videos by ratio closest to 1.0 ──
    all_results = find_high_confidence_neurons(
        model, data, stimuli, activity,
        dataset_dir           = args.dataset_dir,
        stimulus_size         = args.stimulus_size,
        top_neurons           = args.top_neurons,
        min_actual_percentile = args.min_actual_percentile,
        device                = args.device,
    )

    n_total = len(all_results)
    print(f"\n  Saving figures for all {n_total} videos...")

    # ── Step 4: Grad-CAM + two figures per video, rank in filename ──
    for rank, result in enumerate(all_results, start=1):
        stim = result["stim_name"]
        top_nrn = result["top_neurons"]

        print(f"\n  [{rank}/{n_total}]  {stim}  "
              f"best_ratio={result['best_ratio']:.4f}  "
              f"|log|={result['best_log_ratio']:.4f}")

        video_rgb = load_video_rgb_for_display(result["video_path"], args.stimulus_size)

        cams         = []
        pred_rates   = []
        actual_rates = []
        ratios       = []
        log_ratios   = []

        for nidx in top_nrn:
            cam = compute_gradcam(model, result["video_tensor"], int(nidx))
            cams.append(cam) # (T, H_cam, W_cam)
            pred_rates.append(float(result["pred_raw"][nidx]))
            actual_rates.append(float(result["actual_raw"][nidx]))
            ratios.append(float(result["ratio"][nidx]))
            log_ratios.append(float(result["log_ratio"][nidx]))

        pred_arr   = np.array(pred_rates)
        actual_arr = np.array(actual_rates)
        ratios_arr = np.array(ratios)
        lr_arr     = np.array(log_ratios)
        rank_str   = f"rank{rank:04d}"

        plot_population_heatmap(
            video_rgb, cams, top_nrn,
            pred_arr, actual_arr, ratios_arr, lr_arr,
            stim, args.model, rank,
            os.path.join(args.out_dir, f"{rank_str}_{stim}_population.png")
        )

        n_show = len(top_nrn) if args.n_individual == 0 else args.n_individual
        plot_individual_neurons(
            video_rgb, cams, top_nrn,
            pred_arr, actual_arr, ratios_arr, lr_arr,
            stim, args.model, rank,
            os.path.join(args.out_dir, f"{rank_str}_{stim}_individuals.png"),
            n_show=n_show,
        )

    print(f"\n  Done. {n_total * 2} figures saved to '{args.out_dir}/'")
    print(f"  Files named rank0001_... through rank{n_total:04d}_...")
    print(f"  rank0001 = video where model was most perfectly calibrated")
