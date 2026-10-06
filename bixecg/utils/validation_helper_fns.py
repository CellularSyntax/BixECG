import os
import json
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import torch
import torch.nn.functional as F
from torchinfo import summary

from scipy.signal import butter, filtfilt
from scipy.spatial import cKDTree
from scipy import stats as scipy_stats

from sklearn.metrics import r2_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                                 roc_curve, auc, cohen_kappa_score)

from bixecg.utils.helper_fns import (plot_sequence_shaded, build_model, apply_classwise_tolerance_matching)
from bixecg.utils.helper_fns import get_device

def get_tolerance_map():
    return {
        1: 52,    # P-wave ±13 → ~52 ms
        2: 48,    # QRS ±12 → ~48 ms
        3: 124,   # T-wave ±31 → ~124 ms
    }

def majority_vote_smoothing(predictions, window_size=5):
    """
    Apply majority voting smoothing along the sequence.
    Args:
        predictions: np.ndarray of shape (N, seq_len)
        window_size: odd int, number of points to consider for majority voting
    Returns:
        smoothed_predictions: np.ndarray of same shape
    """
    pad = window_size // 2
    padded = np.pad(predictions, ((0,0), (pad, pad)), mode='edge')
    smoothed = np.zeros_like(predictions)

    for i in range(predictions.shape[1]):
        window = padded[:, i:i+window_size]
        smoothed[:, i] = scipy_stats.mode(window, axis=1, keepdims=False)[0]

    return smoothed

def get_model_size(filepath):
    return os.path.getsize(filepath) / 1024  # KB

# ========== Utility Functions ==========
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(signal, lowcut=0.5, highcut=50, fs=250, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, signal)

def measure_inference_time(model, dataloader, device="cuda"):
    model.eval()
    total_time = 0.0
    total_batches = 0
    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            start = time.time()
            _ = model(x_batch)
            torch.cuda.synchronize() if device == "cuda" else None
            end = time.time()
            total_time += (end - start)
            total_batches += 1
    return total_time / total_batches

def plot_bland_altman_ms(a_ms, b_ms, title, save_path_prefix):
    means = (a_ms + b_ms) / 2
    diffs = a_ms - b_ms
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    upper = mean_diff + 1.96 * std_diff
    lower = mean_diff - 1.96 * std_diff

    plt.figure(figsize=(8, 5))
    plt.scatter(means, diffs, alpha=0.5)
    plt.axhline(mean_diff, color='gray', linestyle='--', label=f'Mean Diff = {mean_diff:.2f} ms')
    plt.axhline(upper, color='red', linestyle='--', label=f'+1.96 SD = {upper:.2f} ms')
    plt.axhline(lower, color='blue', linestyle='--', label=f'-1.96 SD = {lower:.2f} ms')
    plt.ylim(lower*2, upper*2)  # Adjust y-limits for better visibility
    plt.xlabel("Mean Time (ms)")
    plt.ylabel("Difference (Pred - True) (ms)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_prefix + ".png")
    plt.savefig(save_path_prefix + ".svg")
    plt.close()

def plot_bland_altman_overlay_colored(overlay_means_dict, overlay_diffs_dict, title, save_path_prefix):
    plt.figure(figsize=(9, 6))
    colors = cm.tab10(np.linspace(0, 1, len(overlay_means_dict)))

    all_means, all_diffs = [], []

    for i, (class_name, means) in enumerate(overlay_means_dict.items()):
        diffs = overlay_diffs_dict[class_name]
        all_means.extend(means)
        all_diffs.extend(diffs)
        plt.scatter(means, diffs, alpha=0.5, label=class_name, color=colors[i])

    all_means = np.array(all_means)
    all_diffs = np.array(all_diffs)
    mean_diff = np.mean(all_diffs)
    std_diff = np.std(all_diffs)
    upper = mean_diff + 1.96 * std_diff
    lower = mean_diff - 1.96 * std_diff

    plt.axhline(mean_diff, color='gray', linestyle='--', label=f'Mean Diff = {mean_diff:.2f} ms')
    plt.axhline(upper, color='red', linestyle='--', label=f'+1.96 SD = {upper:.2f} ms')
    plt.axhline(lower, color='blue', linestyle='--', label=f'-1.96 SD = {lower:.2f} ms')
    plt.ylim(lower*2, upper*2)  # Adjust y-limits for better visibility
    plt.xlabel("Mean Time (ms)")
    plt.ylabel("Difference (Pred - True) (ms)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_prefix + ".png")
    plt.savefig(save_path_prefix + ".svg")
    plt.close()

def analyze_bland_altman_and_correlation_per_sequence(y_true_seq, y_pred_seq, target_names, Fs, save_dir):
    bland_corr_results = {}
    matching_failures = {}
    sample_to_ms = lambda x: (x / Fs) * 1000

    overlay_onset_means_dict = {}
    overlay_onset_diffs_dict = {}
    overlay_offset_means_dict = {}
    overlay_offset_diffs_dict = {}

    for class_id, class_name in enumerate(target_names):
        class_results = {}
        failures = {}

        true_onsets_all, pred_onsets_all = [], []
        true_offsets_all, pred_offsets_all = [], []

        for seq_true, seq_pred in zip(y_true_seq, y_pred_seq):
            transitions_true = np.diff((seq_true == class_id).astype(int))
            transitions_pred = np.diff((seq_pred == class_id).astype(int))

            true_onsets = np.where(transitions_true == 1)[0]
            pred_onsets = np.where(transitions_pred == 1)[0]
            true_offsets = np.where(transitions_true == -1)[0]
            pred_offsets = np.where(transitions_pred == -1)[0]

            true_onsets_all.append(true_onsets)
            pred_onsets_all.append(pred_onsets)
            true_offsets_all.append(true_offsets)
            pred_offsets_all.append(pred_offsets)

        true_onsets_flat = np.concatenate(true_onsets_all)
        pred_onsets_flat = np.concatenate(pred_onsets_all)
        true_offsets_flat = np.concatenate(true_offsets_all)
        pred_offsets_flat = np.concatenate(pred_offsets_all)

        # Match Onsets
        if len(true_onsets_flat) >= 5 and len(pred_onsets_flat) >= 5:
            tree = cKDTree(pred_onsets_flat[:, np.newaxis])
            dists, indices = tree.query(true_onsets_flat[:, np.newaxis], distance_upper_bound=Fs*0.15)

            matched_true = true_onsets_flat[dists != np.inf]
            matched_pred = pred_onsets_flat[indices[dists != np.inf]]

            failures["unmatched_onsets"] = int(np.sum(dists == np.inf))

            pred_ms = sample_to_ms(matched_pred)
            true_ms = sample_to_ms(matched_true)
            diffs = pred_ms - true_ms

            r, p_value = scipy_stats.pearsonr(pred_ms, true_ms)
            r2 = r2_score(true_ms, pred_ms)

            class_results["onset"] = {
                "mean_diff_ms": float(np.mean(diffs)),
                "std_diff_ms": float(np.std(diffs)),
                "pearson_r": float(r),
                "r_squared": float(r2),
                "n_matched": int(len(matched_true))
            }

            overlay_onset_means_dict[class_name] = list((pred_ms + true_ms) / 2)
            overlay_onset_diffs_dict[class_name] = list(diffs)

            plot_bland_altman_ms(
                pred_ms, true_ms,
                title=f"Bland-Altman (Onsets) - {class_name}",
                save_path_prefix=os.path.join(save_dir, f"bland_altman_onsets_{class_name}")
            )
        else:
            failures["onset"] = "Too few onsets"

        # Match Offsets
        if len(true_offsets_flat) >= 5 and len(pred_offsets_flat) >= 5:
            tree = cKDTree(pred_offsets_flat[:, np.newaxis])
            dists, indices = tree.query(true_offsets_flat[:, np.newaxis], distance_upper_bound=Fs*0.15)

            matched_true = true_offsets_flat[dists != np.inf]
            matched_pred = pred_offsets_flat[indices[dists != np.inf]]

            failures["unmatched_offsets"] = int(np.sum(dists == np.inf))

            pred_ms = sample_to_ms(matched_pred)
            true_ms = sample_to_ms(matched_true)
            diffs = pred_ms - true_ms

            r, p_value = scipy_stats.pearsonr(pred_ms, true_ms)
            r2 = r2_score(true_ms, pred_ms)

            class_results["offset"] = {
                "mean_diff_ms": float(np.mean(diffs)),
                "std_diff_ms": float(np.std(diffs)),
                "pearson_r": float(r),
                "r_squared": float(r2),
                "n_matched": int(len(matched_true))
            }

            overlay_offset_means_dict[class_name] = list((pred_ms + true_ms) / 2)
            overlay_offset_diffs_dict[class_name] = list(diffs)

            plot_bland_altman_ms(
                pred_ms, true_ms,
                title=f"Bland-Altman (Offsets) - {class_name}",
                save_path_prefix=os.path.join(save_dir, f"bland_altman_offsets_{class_name}")
            )
        else:
            failures["offset"] = "Too few offsets"

        bland_corr_results[class_name] = class_results
        matching_failures[class_name] = failures

    if overlay_onset_means_dict:
        plot_bland_altman_overlay_colored(
            overlay_onset_means_dict, overlay_onset_diffs_dict,
            title="Overlay Bland-Altman (Onsets)",
            save_path_prefix=os.path.join(save_dir, "overlay_bland_altman_onsets")
        )

    if overlay_offset_means_dict:
        plot_bland_altman_overlay_colored(
            overlay_offset_means_dict, overlay_offset_diffs_dict,
            title="Overlay Bland-Altman (Offsets)",
            save_path_prefix=os.path.join(save_dir, "overlay_bland_altman_offsets")
        )

    return bland_corr_results, matching_failures

def fit_temperature(model, logits, true_labels, device="cuda"):
    T = torch.ones(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / T, true_labels, reduction='mean')
        loss.backward()
        return loss

    optimizer.step(eval)
    return T.item()

def compute_ece_bootstrapped(y_true, probs, n_bins=20, n_bootstrap=200):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])

    def compute_single_ece(y_true_batch, confidences_batch, predictions_batch):
        accs, confs, sizes = [], [], []
        for i in range(n_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i+1]
            mask = (confidences_batch > lower) & (confidences_batch <= upper)
            if np.any(mask):
                accs.append(np.mean(predictions_batch[mask] == y_true_batch[mask]))
                confs.append(np.mean(confidences_batch[mask]))
                sizes.append(np.sum(mask))
            else:
                accs.append(0.0)
                confs.append(0.0)
                sizes.append(0)
        accs = np.array(accs)
        confs = np.array(confs)
        sizes = np.array(sizes)
        ece = np.sum(np.abs(accs - confs) * (sizes / np.sum(sizes)))
        return ece, accs, confs

    # Compute on full set
    ece, accs, confs = compute_single_ece(y_true, confidences, predictions)

    # Bootstrap
    ece_bootstrap = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        ece_b, _, _ = compute_single_ece(y_true[indices], confidences[indices], predictions[indices])
        ece_bootstrap.append(ece_b)

    lower = np.percentile(ece_bootstrap, 2.5)
    upper = np.percentile(ece_bootstrap, 97.5)

    return ece, accs, confs, bin_boundaries, (lower, upper)

def plot_reliability_diagram(accs, confs, bin_boundaries, ece_value, save_path, ci=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot bars manually for accuracies
    ax.bar(bin_centers, accs, width=(bin_boundaries[1] - bin_boundaries[0]) * 0.9, 
           color='skyblue', edgecolor='black', label="Accuracy")

    # Plot confidence as scatter points
    ax.scatter(bin_centers, confs, color='red', s=50, label='Confidence')

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)

    title = f"Reliability Diagram\nECE = {ece_value:.3f}"
    if ci is not None:
        title += f" (95% CI: {ci[0]:.3f}-{ci[1]:.3f})"
    ax.set_title(title)

    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Actual Accuracy")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    fig.savefig(save_path + ".png")
    fig.savefig(save_path + ".svg")
    plt.close(fig)

# ========= Saliency Map =========
def compute_saliency_maps(model, inputs, target_labels, device):
    # Save original training state
    was_training = model.training
    
    # Set model to train() because sLSTM requires it for backward()
    model.train()

    inputs = inputs.clone().detach().to(device)
    inputs.requires_grad_(True)

    outputs = model(inputs)
    loss = F.cross_entropy(outputs.view(-1, outputs.shape[-1]), target_labels.view(-1), ignore_index=-100)
    
    model.zero_grad()
    loss.backward()
    
    saliency = inputs.grad.abs().detach().cpu().numpy()

    # Restore original state
    if not was_training:
        model.eval()
        
    return saliency

def run_mc_dropout_uncertainty(model, inputs, n_samples=20, batch_size=32):
    model.train()  # enable dropout
    device = next(model.parameters()).device
    preds = []

    inputs = inputs.to(device)
    dataset = torch.utils.data.TensorDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\n🔄 Starting MC Dropout with {n_samples} samples, batch size {batch_size}...")
    for sample_idx in range(n_samples):
        all_probs = []
        print(f"  📦 MC sample {sample_idx+1}/{n_samples}...")
        for batch_idx, (x_batch,) in enumerate(loader):
            x_batch = x_batch.to(device)
            with torch.no_grad():
                outputs = model(x_batch)
                probs = torch.softmax(outputs, dim=-1)
                all_probs.append(probs.cpu().numpy())
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(f"    ✅ Batch {batch_idx+1}/{len(loader)} completed")
        all_probs = np.concatenate(all_probs, axis=0)  # shape (N, T, C)
        preds.append(all_probs)

    preds = np.stack(preds)  # (n_samples, N, T, C)
    mean_preds = np.mean(preds, axis=0)
    entropy = -np.sum(mean_preds * np.log(mean_preds + 1e-8), axis=-1)  # (N, T)

    print("✅ MC Dropout Uncertainty computation finished.\n")
    return entropy

def quantify_classwise_uncertainty(model, inputs, true_labels, target_names, device="cuda", n_samples=20, batch_size=16, save_dir=None):
    model.train()  # Enable dropout during inference
    inputs = inputs.to(device)
    true_labels = true_labels.to(device)

    preds = []
    dataset = torch.utils.data.TensorDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for _ in range(n_samples):
        all_probs = []
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            with torch.no_grad():
                outputs = model(x_batch)
                probs = torch.softmax(outputs, dim=-1)
                all_probs.append(probs.cpu().numpy())
        all_probs = np.concatenate(all_probs, axis=0)
        preds.append(all_probs)

    preds = np.stack(preds)  # (n_samples, N, T, C)
    mean_preds = np.mean(preds, axis=0)  # (N, T, C)
    entropy = -np.sum(mean_preds * np.log(mean_preds + 1e-8), axis=-1)  # (N, T)

    # Flatten entropy and labels
    entropy_flat = entropy.flatten()
    labels_flat = true_labels.cpu().numpy().flatten()

    # Mask out padding
    mask = labels_flat != -100
    entropy_flat = entropy_flat[mask]
    labels_flat = labels_flat[mask]

    # Create DataFrame
    df = pd.DataFrame({
        "Entropy": entropy_flat,
        "Class": [target_names[lbl] for lbl in labels_flat]
    })

    # --- Violin plot ---
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x="Class", y="Entropy", inner="box", cut=0)
    plt.title("Predictive Uncertainty per Class (MC Dropout)")
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "classwise_uncertainty_violinplot.png"), dpi=300)
        plt.savefig(os.path.join(save_dir, "classwise_uncertainty_violinplot.svg"))
    else:
        plt.show()
    plt.close()

    # Print average uncertainty
    class_means = df.groupby("Class")["Entropy"].mean()
    print("\n🎻 Average Uncertainty (Entropy) per Class:")
    print(class_means)

    return df, class_means

def plot_uncertainty_histograms(entropy_values, save_path):
    plt.figure(figsize=(6, 5))
    plt.hist(entropy_values.flatten(), bins=50, color='purple', alpha=0.7)
    plt.title("Predictive Entropy Histogram")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + ".png")
    plt.savefig(save_path + ".svg")
    plt.close()

def plot_saliency_overlay(input_signal, saliency, save_path_prefix, fs=250, pred_labels=None, true_labels=None):
    """
    ECG plotted directly with saliency coloring (magma colormap).
    Ground-truth and predicted labels shown as two thin ribbons (flare colormap).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.colors import to_rgba

    sns.set_style("ticks")
    sns.set_context("notebook", rc={
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    fig, ax = plt.subplots(figsize=(12, 4))

    time = np.linspace(0, len(input_signal) / fs, len(input_signal))

    # Normalize saliency
    saliency_norm = (saliency.squeeze() - np.min(saliency.squeeze())) / (np.max(saliency.squeeze()) - np.min(saliency.squeeze()) + 1e-8)

    # Colormaps
    cmap_saliency = sns.color_palette("magma", as_cmap=True)
    cmap_labels = sns.color_palette("flare", as_cmap=True)

    # Define ribbons
    signal_min = np.min(input_signal)
    signal_max = np.max(input_signal)
    signal_range = signal_max - signal_min

    truth_bottom = signal_min - 0.25 * signal_range
    truth_top = signal_min - 0.20 * signal_range
    pred_bottom = signal_min - 0.19 * signal_range
    pred_top = signal_min - 0.14 * signal_range

    # Helper to lighten colors
    def lighten_color(color, factor=0.5):
        r, g, b, a = to_rgba(color)
        return (r + (1 - r) * factor,
                g + (1 - g) * factor,
                b + (1 - b) * factor,
                a)

    # Find number of classes
    n_classes = 1
    if true_labels is not None:
        n_classes = max(n_classes, np.max(true_labels))
    if pred_labels is not None:
        n_classes = max(n_classes, np.max(pred_labels))
    n_classes += 1 if n_classes > 0 else 0

    # === Ground truth ribbon ===
    if true_labels is not None:
        true_labels = np.asarray(true_labels)
        for i in range(len(time) - 1):
            label = true_labels[i]
            if label in [-100, 0] or np.isnan(label):
                continue
            color = cmap_labels(label / max(n_classes-1, 1))
            color_light = lighten_color(color, factor=0.4)
            ax.fill_between(
                time[i:i+2],
                truth_bottom, truth_top,
                color=color_light,
                edgecolor=None,
                linewidth=0,
                zorder=0
            )

    # === Thin white separator between ribbons
    ax.plot(time, np.full_like(time, signal_min - 0.195 * signal_range),
            color='white', linewidth=1.5, zorder=1)

    # === Prediction ribbon ===
    if pred_labels is not None:
        pred_labels = np.asarray(pred_labels)
        for i in range(len(time) - 1):
            label = pred_labels[i]
            if label in [-100, 0] or np.isnan(label):
                continue
            color = cmap_labels(label / max(n_classes-1, 1))
            ax.fill_between(
                time[i:i+2],
                pred_bottom, pred_top,
                color=color,
                edgecolor=None,
                linewidth=0,
                zorder=2
            )

    # === ECG plotted with saliency coloring ===
    for i in range(len(time) - 1):
        ax.plot(
            time[i:i+2],
            input_signal.squeeze()[i:i+2],
            color=cmap_saliency(saliency_norm[i]),
            linewidth=2,
            zorder=3
        )

    # === Correct colorbar creation for saliency
    sm = plt.cm.ScalarMappable(cmap=cmap_saliency, norm=plt.Normalize(vmin=np.min(saliency), vmax=np.max(saliency)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Saliency (a.u.)')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (n.u.)")
    ax.grid(False)
    fig.tight_layout()
    sns.despine()

    fig.savefig(save_path_prefix + ".png", dpi=300)
    fig.savefig(save_path_prefix + ".svg")
    plt.close(fig)


def plot_average_saliency(model, inputs, targets, device, save_path_prefix, n_samples=100):
    import matplotlib.pyplot as plt

    model.eval()
    indices = random.sample(range(len(inputs)), min(n_samples, len(inputs)))

    accumulated_saliency = None

    for idx in indices:
        x_single = inputs[idx:idx+1].to(device)
        y_single = targets[idx:idx+1].to(device)

        saliency = compute_saliency_maps(model, x_single, y_single, device)
        padding_mask = (y_single != -100)  # y_single is your labels

        saliency = saliency * padding_mask.float().unsqueeze(-1)  # Set saliency at padding to 0

        if accumulated_saliency is None:
            accumulated_saliency = saliency
        else:
            accumulated_saliency += saliency

    avg_saliency = accumulated_saliency.squeeze() / len(indices)

    plt.figure(figsize=(10, 4))
    plt.plot(avg_saliency, label="Average Saliency", color="red")
    plt.title(f"Average Saliency Over {len(indices)} Sequences")
    plt.xlabel("Sample Index")
    plt.ylabel("Average Saliency")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path_prefix + ".png")
    plt.savefig(save_path_prefix + ".svg")
    plt.close()

def plot_classwise_uncertainty(entropy, predicted_classes, target_names, save_path_prefix):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 5))

    for class_idx, class_name in enumerate(target_names):
        class_entropy = entropy[predicted_classes == class_idx]
        sns.kdeplot(class_entropy, label=class_name, fill=True, alpha=0.3)

    plt.xlabel("Entropy (Uncertainty)")
    plt.ylabel("Density")
    plt.title("Classwise Uncertainty Distribution (MC Dropout)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path_prefix + ".png")
    plt.savefig(save_path_prefix + ".svg")
    plt.close()

def quantify_wave_vs_background_uncertainty(model, inputs, true_labels, wave_class_indices, device="cuda", n_samples=20, batch_size=16, save_dir=None):
    import pandas as pd
    import seaborn as sns
    import os
    import matplotlib.pyplot as plt

    model.train()  # Enable MC Dropout
    inputs = inputs.to(device)
    true_labels = true_labels.to(device)

    preds = []
    dataset = torch.utils.data.TensorDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for _ in range(n_samples):
        all_probs = []
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            with torch.no_grad():
                outputs = model(x_batch)
                probs = torch.softmax(outputs, dim=-1)
                all_probs.append(probs.cpu().numpy())
        all_probs = np.concatenate(all_probs, axis=0)
        preds.append(all_probs)

    preds = np.stack(preds)  # (n_samples, N, T, C)
    mean_preds = np.mean(preds, axis=0)  # (N, T, C)
    entropy = -np.sum(mean_preds * np.log(mean_preds + 1e-8), axis=-1)  # (N, T)

    entropy_flat = entropy.flatten()
    labels_flat = true_labels.cpu().numpy().flatten()

    mask = labels_flat != -100
    entropy_flat = entropy_flat[mask]
    labels_flat = labels_flat[mask]

    # Label as 'Wave' or 'Background'
    is_wave = np.isin(labels_flat, wave_class_indices)
    labels_binary = np.where(is_wave, "Wave", "Background")

    # Create dataframe
    df = pd.DataFrame({
        "Entropy": entropy_flat,
        "Region": labels_binary
    })

    # --- Boxplot ---
    plt.figure(figsize=(7, 5))
    sns.violinplot(data=df, x="Region", y="Entropy", hue="Region", palette="Set2", inner="box", cut=0, legend=False)
    plt.title("Uncertainty for Waves vs Background (MC Dropout)")
    plt.grid(True, axis='y')
    plt.tight_layout()


    if save_dir:
        plt.savefig(os.path.join(save_dir, "uncertainty_wave_vs_background.png"), dpi=300)
        plt.savefig(os.path.join(save_dir, "uncertainty_wave_vs_background.svg"))
    else:
        plt.show()
    plt.close()

    # --- Summary Stats ---
    mean_wave = df[df["Region"] == "Wave"]["Entropy"].mean()
    mean_background = df[df["Region"] == "Background"]["Entropy"].mean()
    ratio = mean_background / mean_wave if mean_wave > 0 else np.inf

    print("\n📊 Uncertainty Summary:")
    print(f"  Mean Wave Entropy: {mean_wave:.4f}")
    print(f"  Mean Background Entropy: {mean_background:.4f}")
    print(f"  Ratio (Background / Wave): {ratio:.2f}x")

    # Save JSON
    if save_dir:
        summary = {
            "mean_wave_entropy": float(mean_wave),
            "mean_background_entropy": float(mean_background),
            "background_to_wave_ratio": float(ratio)
        }
        with open(os.path.join(save_dir, "uncertainty_wave_vs_background_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

    return df, mean_wave, mean_background, ratio


# ========= Master Explainability Launcher =========
def explain_model_predictions(model, ludb_x, ludb_y, save_dir, target_names, SEQ_LEN, fs=250):
    print("\n🔍 Starting explainability analysis...")

    device = next(model.parameters()).device
    model.eval()

    explain_dir = os.path.join(save_dir, "explainability")
    os.makedirs(explain_dir, exist_ok=True)

    # --- Pick a few random examples for explainability
    indices = random.sample(range(len(ludb_x)), 5)

    for idx in indices:
        x_single = ludb_x[idx:idx+1].to(device)
        y_single = ludb_y[idx:idx+1].to(device)

        # === Saliency Map
        saliency = compute_saliency_maps(model, x_single.clone(), y_single.clone(), device)
        # Mask padded regions before plotting
        padding_mask = (y_single != -100)  # y_single is your labels

        saliency = torch.tensor(saliency, device=padding_mask.device)  # 💥 convert saliency to tensor
        saliency = saliency * padding_mask.float().unsqueeze(-1)
        saliency = saliency.detach().cpu().numpy()  # (optional) if later plotting expects numpy

        with torch.no_grad():
            logits_single = model(x_single)
            pred_labels_single = torch.argmax(logits_single, dim=-1)  # (1, T)
            

        pred_labels_single_smoothed = majority_vote_smoothing(
            pred_labels_single.cpu().numpy(), window_size=30
        )
        plot_saliency_overlay(
            input_signal=x_single[0].detach().cpu().numpy(),
            saliency=saliency[0],
            save_path_prefix=os.path.join(explain_dir, f"saliency_overlay_{idx}"),
            pred_labels=pred_labels_single_smoothed[0],
            true_labels=y_single[0].detach().cpu().numpy() 
        )

    # === Uncertainty Estimation (MC Dropout, fast version) on subset
    print("🧮 Computing MC Dropout uncertainties (fast version)...")

    subset_size = 20
    n_samples = 5
    batch_size_mc = 8

    torch.cuda.empty_cache()

    subset_indices = np.random.choice(len(ludb_x), subset_size, replace=False)
    subset_x = ludb_x[subset_indices]

    entropy_map = run_mc_dropout_uncertainty(
        model=model,
        inputs=subset_x.to(device),
        n_samples=n_samples,
        batch_size=batch_size_mc
    )

    plot_uncertainty_histograms(entropy_map, save_path=os.path.join(explain_dir, "uncertainty_histogram"))

    print("\n📊 Quantifying classwise uncertainty...")
    quantify_classwise_uncertainty(
        model=model,
        inputs=ludb_x,
        true_labels=ludb_y,
        target_names=target_names,
        device=device,
        n_samples=20,
        batch_size=16,
        save_dir=explain_dir
    )

    print("\n📊 Quantifying wave vs background uncertainty...")
    wave_class_indices = [1, 2, 3] 
    quantify_wave_vs_background_uncertainty(
        model=model,
        inputs=ludb_x,
        true_labels=ludb_y,
        wave_class_indices=wave_class_indices,
        device=device,
        n_samples=20,
        batch_size=16,
        save_dir=explain_dir
    )

    print("✅ Explainability analysis completed.")


def plot_entropy_vs_error_rate(y_true, probs, entropy, save_path_prefix):
    sns.set_style("whitegrid")
    
    preds = np.argmax(probs, axis=1)
    correctness = (preds == y_true).astype(int)  # 1 = correct, 0 = wrong

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=entropy, y=1 - correctness, alpha=0.6, s=20, edgecolor=None)

    plt.title("Entropy vs. Error Rate", fontsize=14)
    plt.xlabel("Predictive Entropy", fontsize=12)
    plt.ylabel("Error (1 = Wrong, 0 = Correct)", fontsize=12)
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()

    plt.savefig(save_path_prefix + ".png", dpi=300)
    plt.savefig(save_path_prefix + ".svg")
    plt.close()

def plot_entropy_vs_confidence(probs, entropy, save_path_prefix):
    sns.set_style("whitegrid")
    
    confidences = np.max(probs, axis=1)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=confidences, y=entropy, alpha=0.6, s=20, edgecolor=None)

    plt.title("Entropy vs. Confidence", fontsize=14)
    plt.xlabel("Predicted Confidence", fontsize=12)
    plt.ylabel("Predictive Entropy", fontsize=12)
    plt.grid(True)
    plt.xlim(0, 1.05)
    plt.tight_layout()

    plt.savefig(save_path_prefix + ".png", dpi=300)
    plt.savefig(save_path_prefix + ".svg")
    plt.close()

def run_inference(model, test_loader, ignore_index=-100):
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_trues, all_logits = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            logits[y_batch == ignore_index] = float('-inf')
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_trues), np.concatenate(all_logits, axis=0)

def evaluate_strict(y_true, y_pred, target_names, save_dir, ignore_index=-100):
    print("\n📊 Strict Evaluation:")
    labels = list(range(len(target_names)))
    mask = y_true != ignore_index

    report = classification_report(
        y_true[mask], y_pred[mask], labels=labels,
        target_names=target_names, digits=3, output_dict=True
    )
    kappa = cohen_kappa_score(y_true[mask], y_pred[mask])

    print(json.dumps(report, indent=4))
    print(f"📈 Cohen's Kappa (Strict): {kappa:.4f}")

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true[mask], y_pred[mask], labels=labels, 
        display_labels=target_names, normalize='true', cmap='Blues', ax=ax
    )
    ax.grid(False)
    ax.set_title("Strict Confusion Matrix (Normalized)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "confusion_matrix_strict.png"))
    fig.savefig(os.path.join(save_dir, "confusion_matrix_strict.svg"))
    plt.close(fig)

    return {"classification_report": report, "cohen_kappa": kappa}

def generate_roc_and_temperature_scaling(model, logits_all, y_true, save_dir, target_names, ignore_index=-100):
    print("\n🌡️ Fitting temperature scaling and generating ROC curves...")
    device = next(model.parameters()).device
    labels = list(range(len(target_names)))

    y_true_flat = y_true.flatten()
    valid_mask = y_true_flat != ignore_index
    y_true_valid = y_true_flat[valid_mask]
    y_score_flat = logits_all.reshape(-1, logits_all.shape[-1])
    y_score_valid = y_score_flat[valid_mask]

    y_true_bin = label_binarize(y_true_valid, classes=labels)

    valid_logits_tensor = torch.tensor(y_score_valid, dtype=torch.float32, device=device)
    valid_labels_tensor = torch.tensor(y_true_valid, dtype=torch.long, device=device)
    optimal_temperature = fit_temperature(model, valid_logits_tensor, valid_labels_tensor, device=device)

    y_score_valid_scaled = valid_logits_tensor / optimal_temperature
    probs = torch.softmax(y_score_valid_scaled, dim=1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    auc_values = []
    for i, class_name in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

    macro_auc = np.mean(auc_values)
    print(f"📈 Macro AUC (Strict, Temp Scaled): {macro_auc:.4f}")

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title("ROC Curves (Strict Evaluation, Temp Scaled)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "roc_curves_strict.png"))
    fig.savefig(os.path.join(save_dir, "roc_curves_strict.svg"))
    plt.close(fig)

    return probs, optimal_temperature, macro_auc

def save_random_shaded_sequences(x_test_tensor, y_test_tensor, y_pred_seq, save_dir, SEQ_LEN, N_examples=20):
    """
    Save random shaded ECG sequences showing ground truth and model predictions.
    """
    shaded_dir = os.path.join(save_dir, "shaded_sequences")
    os.makedirs(shaded_dir, exist_ok=True)
    
    random_indices = np.random.choice(len(x_test_tensor), size=N_examples, replace=False)

    for idx in random_indices:
        ecg_signal = x_test_tensor[idx].cpu().numpy().squeeze()
        true_labels = y_test_tensor[idx].cpu().numpy()
        pred_labels = y_pred_seq[idx]

        # Majority vote smoothing is already applied before calling this function
        fig = plot_sequence_shaded(
            ecg_signal=ecg_signal,
            true_labels=true_labels,
            raw_preds=pred_labels,
            SEQ_LEN=SEQ_LEN
        )

        fig_path_png = os.path.join(shaded_dir, f"sequence_{idx}.png")
        fig_path_svg = os.path.join(shaded_dir, f"sequence_{idx}.svg")
        fig.savefig(fig_path_png)
        fig.savefig(fig_path_svg)
        plt.close(fig)

def evluate_on_db(model, test_loader, x_test_tensor, y_test_tensor,
                     SEQ_LEN, save_dir, Fs=250, target_names=None, tolerance_map=None,
                     ignore_index=-100, N_examples=20, loader=None, model_path=None):
    print("\n🚀 Starting evaluation...")
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 Results will be saved to: {save_dir}")

    stats = {}
    
    # 1. Inference
    y_pred, y_true, logits_all = run_inference(model, test_loader, ignore_index)
    y_pred_seq = y_pred.reshape(-1, SEQ_LEN)
    y_true_seq = y_true.reshape(-1, SEQ_LEN)
    y_pred_seq = majority_vote_smoothing(y_pred_seq, window_size=30)

    # 2. Strict Evaluation
    strict_stats = evaluate_strict(y_true, y_pred, target_names, save_dir, ignore_index)
    stats.update({"strict_classification_report": strict_stats["classification_report"],
                  "cohen_kappa_strict": strict_stats["cohen_kappa"]})

    # 3. ROC + Temperature Scaling
    probs, temp, macro_auc = generate_roc_and_temperature_scaling(
        model, logits_all, y_true, save_dir, target_names, ignore_index
    )
    stats.update({"temperature_scaling_T": temp, "strict_macro_auc": macro_auc})

    # 4. Calibration
    ece, accs, confs, bins, ece_ci = compute_ece_bootstrapped(y_true[y_true != ignore_index], probs, n_bins=20, n_bootstrap=200)
    stats["expected_calibration_error_overall"] = {"ece": ece, "95%_CI": [ece_ci[0], ece_ci[1]]}
    plot_reliability_diagram(accs, confs, bins, ece_value=ece, ci=ece_ci,
                              save_path=os.path.join(save_dir, "reliability_diagram_overall"))

    # 5. Uncertainty
    subset_indices = np.random.choice(len(x_test_tensor), size=20, replace=False)
    subset_x = x_test_tensor[subset_indices]
    subset_y = y_test_tensor[subset_indices]
    entropy = run_mc_dropout_uncertainty(model, subset_x.to(next(model.parameters()).device), n_samples=10, batch_size=256)
    
    # Additional plots: entropy vs error/confidence
    with torch.no_grad():
        subset_logits = model(subset_x.to(next(model.parameters()).device))
        subset_logits[subset_y == ignore_index] = float('-inf')
        subset_probs = torch.softmax(subset_logits, dim=-1).cpu().numpy()
    entropy_valid = entropy.flatten()[subset_y.flatten() != ignore_index]
    probs_valid = subset_probs.reshape(-1, subset_probs.shape[-1])[subset_y.flatten() != ignore_index]
    y_true_valid = subset_y.flatten()[subset_y.flatten() != ignore_index].cpu().numpy()

    plot_entropy_vs_error_rate(y_true_valid, probs_valid, entropy_valid,
                               save_path_prefix=os.path.join(save_dir, "entropy_vs_error_rate"))
    plot_entropy_vs_confidence(probs_valid, entropy_valid,
                               save_path_prefix=os.path.join(save_dir, "entropy_vs_confidence"))

    # 6. Bland-Altman and Correlation
    bland_corr_results, matching_failures = analyze_bland_altman_and_correlation_per_sequence(
        y_true_seq=y_true_seq, y_pred_seq=y_pred_seq, target_names=target_names, Fs=Fs, save_dir=save_dir
    )
    stats.update({"bland_altman_correlation": bland_corr_results,
                  "bland_altman_matching_failures": matching_failures})

    # 7. Tolerance-aware evaluation (optional)
    if tolerance_map is not None:
        print("\n📏 Running tolerance-aware evaluation...")
        y_pred_tol = apply_classwise_tolerance_matching(
            y_true_seq, y_pred_seq, fs=Fs,
            tolerance_map=tolerance_map, ignore_index=ignore_index
        )
        y_true_flat = y_true_seq.flatten()
        y_pred_tol_flat = y_pred_tol.flatten()
        valid_mask = y_true_flat != ignore_index

        report_tol = classification_report(
            y_true_flat[valid_mask], y_pred_tol_flat[valid_mask],
            labels=list(range(len(target_names))), target_names=target_names, digits=3, output_dict=True
        )
        tol_kappa = cohen_kappa_score(y_true_flat[valid_mask], y_pred_tol_flat[valid_mask])
        stats.update({"tolerance_aware_classification_report": report_tol,
                      "cohen_kappa_tolerance": tol_kappa})
        
        # Tolerance Confusion Matrix
        labels = list(range(len(target_names)))
        fig_cm_tol, ax_cm_tol = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_true_flat[valid_mask], y_pred_tol_flat[valid_mask], labels=labels,
            display_labels=target_names, normalize='true', cmap='Blues', ax=ax_cm_tol
        )
        ax_cm_tol.grid(False)
        ax_cm_tol.set_title("Tolerance-aware Confusion Matrix (Normalized)")
        fig_cm_tol.tight_layout()
        fig_cm_tol.savefig(os.path.join(save_dir, "confusion_matrix_tolerance.png"))
        fig_cm_tol.savefig(os.path.join(save_dir, "confusion_matrix_tolerance.svg"))
        plt.close(fig_cm_tol)
        
        # Tolerance ROC Curve
        y_true_bin_tol = label_binarize(y_true_flat[valid_mask], classes=labels)
        fig_roc_tol, ax_roc_tol = plt.subplots(figsize=(8, 6))
        auc_values_tol = []

        for i, class_name in enumerate(target_names):
            fpr, tpr, _ = roc_curve(y_true_bin_tol[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            auc_values_tol.append(roc_auc)
            ax_roc_tol.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

        ax_roc_tol.plot([0, 1], [0, 1], 'k--')
        ax_roc_tol.set_title("ROC Curves (Tolerance-aware, Temp Scaled)")
        ax_roc_tol.set_xlabel("False Positive Rate")
        ax_roc_tol.set_ylabel("True Positive Rate")
        ax_roc_tol.legend(loc="lower right")
        fig_roc_tol.savefig(os.path.join(save_dir, "roc_curves_tolerance.png"))
        fig_roc_tol.savefig(os.path.join(save_dir, "roc_curves_tolerance.svg"))
        plt.close(fig_roc_tol)

    # 8. Explainability
    explain_model_predictions(model, x_test_tensor, y_test_tensor, save_dir, target_names, SEQ_LEN, fs=Fs)

    # 9. Save random shaded sequences
    save_random_shaded_sequences(x_test_tensor, y_test_tensor, y_pred_seq, save_dir, SEQ_LEN, N_examples)

    print("\n✅ Evaluation finished successfully!")

    full_info = {
        "model_size_kb": get_model_size(model_path),
        "inference_time_per_batch_sec": measure_inference_time(model, loader),
        "evaluation_metrics": stats
    }

    # 10. Save plots data for later combined plot generation

    plots_data = {}

    # --- Calibration plot (reliability diagram)
    plots_data["reliability_diagram"] = {
        "accuracies": accs,
        "confidences": confs,
        "bins": bins,
        "ece": ece,
        "ece_ci": ece_ci
    }

    # --- Entropy vs Error/Confidence
    plots_data["entropy_vs_error_confidence"] = {
        "entropy_valid": entropy_valid,
        "probs_valid": probs_valid,
        "y_true_valid": y_true_valid
    }

    # --- Bland-Altman and Correlation
    plots_data["bland_altman_correlation"] = bland_corr_results
    plots_data["bland_altman_matching_failures"] = matching_failures

    return full_info, plots_data

def load_model(model_name):
    model_metadata_path = f"./{model_name}.json"
    model_path = f"./{model_name}.pt"

    # Load metadata
    with open(model_metadata_path, "r") as f:
        metadata = json.load(f)

    config = metadata["config"]

    device = get_device(config)

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    summary(model)

    Fs = config.get('Fs', 250) # Sampling frequency


    return model, config, Fs, model_path

def print_ascii_logo():
    RESET = "\033[0m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Define a palette of ANSI colors to randomly choose from (excluding blue)
    COLOR_PALETTE = [
        "\033[91m",  # Red
        "\033[92m",  # Green
        "\033[93m",  # Yellow
        "\033[95m",  # Magenta
        "\033[96m",  # Cyan
        "\033[90m",  # Bright Black (Gray)
    ]

    raw_logo = r"""                                                                                                                                             
@@@@@@@@   @@@            @@@@@@@@@    @@@@@@@@     @@@@@@@            +-                                 
@@@@@@@@@  @@@            @@@@@@@@@  @@@@@@@@@   @@@@@@@@@@@@         -= =                                 
@@@   @@@       @@@  @@@  @@@        @@@        @@@                   -- =:                                 
@@@@@@@@@  @@@   @@@@@@   @@@@@@@@@ @@@         @@@                   -+  =                                 
@@@    @@@ @@@    @@@@    @@@       @@@         @@@    @@@@@@         *=   =.                               
@@@    @@@ @@@   @@@@@@   @@@        @@@         @@@      @@@         +=   :=                               
@@@@@@@@@  @@@  @@@  @@@  @@@@@@@@@   @@@@@@@@@@  @@@@@@@@@@@   #:    ==    +:                              
                                                               =  @   .*     =:                
--#+=:-.@:=.:*..+:-:*:..+#_*.:-=.#.--+:.=--.*:--:::.*:.:.- # =*   =-  -+      =.   .-@.:*=--:+#-- 
                                                                   +- =+        =:+                      
                                                                    +::+                                     
                                                                     -#-                                      
"""
    colored_logo = ''.join([
        f"{GRAY}@{RESET}" if c == '@' else f"{random.choice(COLOR_PALETTE)}{c}{RESET}"
        for c in raw_logo
    ])

    print(colored_logo)

    print("""
The copyrights of this software are owned by Medical University of Vienna. Please refer to the LICENSE and
README.md files for licensing instructions.The source code can be found at the following GitHub repository:
https://github.com/CellularSyntax/BixECG
""")
    print("---")


if __name__ == "__main__":
    print_ascii_logo()
