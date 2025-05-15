# Standard library
import contextlib
import io
import os
import random
import sys
import time
import datetime
import json
from tqdm import tqdm
# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy.signal import butter, filtfilt
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from scipy.stats.mstats import winsorize
from scipy.signal import resample
import wfdb
import ast
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

# Local imports
from src.models.models import get_model
from src.models.bixecg import BiXLSTM, xLSTM
from src.models.jimenez_cnn import JimenezCNN1D
from src.models.liu_cnn_bilstm import LiuCNNBilstm
from src.models.peimankar_cnn_bilstm import PeimankarCnnBilstm

@contextlib.contextmanager
def redirect_output_to_file(log_path="logs/build_log.txt"):
    """
    Redirects stdout and stderr to a file in the project root, regardless of where the script is run.
    Automatically creates the directory if it does not exist.
    """
    # Resolve to project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    abs_log_path = os.path.join(base_dir, log_path)

    os.makedirs(os.path.dirname(abs_log_path), exist_ok=True)
    with open(abs_log_path, 'w') as logfile:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = logfile, logfile
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal, lowcut=0.5, highcut=50, fs=250, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)

def log_per_class_f1(model: nn.Module, dataloader, epoch, writer, class_names, ignore_index=-100, device=None):
    model.eval()
    all_preds = []
    all_trues = []

    if device == None:
        device = next(model.parameters()).device

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            logits[y_batch == ignore_index] = float('-inf')
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())

    y_true = np.concatenate(all_trues).flatten()
    y_pred = np.concatenate(all_preds).flatten()

    # Mask out padding
    mask = y_true != ignore_index
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    # Per-class F1 scores
    f1s = f1_score(y_true_masked, y_pred_masked, average=None, labels=np.arange(len(class_names)))
    macro_f1 = f1_score(y_true_masked, y_pred_masked, average='macro')

    print(f"\nEpoch {epoch+1:02d} F1 Scores (Masked):")
    for idx, class_name in enumerate(class_names):
        writer.add_scalar(f"F1/{class_name}", f1s[idx], epoch)
        print(f"  F1 ({class_name}): {f1s[idx]:.4f}")

    writer.add_scalar("F1/macro", macro_f1, epoch)
    print(f"  Macro F1: {macro_f1:.4f}")
    return f1s, macro_f1

def evaluate_macro_f1(model, val_loader, device=None):
    model.eval()
    all_preds = []
    all_trues = []
    if device == None:
        device = next(model.parameters()).device
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())
    y_true = np.concatenate(all_trues).flatten()
    y_pred = np.concatenate(all_preds).flatten()
    mask = y_true != -100
    macro_f1 = f1_score(y_true[mask], y_pred[mask], average="macro")
    return macro_f1

def z_normalize(seq):
    return (seq - np.mean(seq)) / (np.std(seq) + 1e-8)

def extract_beat_aligned_sequences(
    ecg_signal, label_signal, patient_signal, lead_signal, fs, 
    beats_per_seq=2, seq_len_seconds=2.0, pad_label=-100, beat_aligned=True
):
    """
    Extract ECG sequences based on either p-wave (label==1) onset positions (beat-aligned) 
    or simple fixed-length segmentation.

    Parameters:
    - ecg_signal: np.ndarray, 1D ECG trace
    - label_signal: np.ndarray, 1D label trace (same length as ecg_signal)
    - patient_signal: np.ndarray, 1D array of patient IDs (same length as ecg_signal)
    - lead_signal: np.ndarray, 1D array of lead numbers (same length as ecg_signal)
    - fs: int, sampling frequency (Hz)
    - beats_per_seq: int, number of p-wave-anchored beats per sequence (only if beat_aligned=True)
    - seq_len_seconds: float, length of each sequence in seconds
    - pad_label: int, label to use for padding (ignored during loss computation)
    - beat_aligned: bool, whether to extract sequences aligned to beats (True) or fixed-length (False)

    Returns:
    - x_seqs: np.ndarray of shape (N, SEQ_LEN)
    - y_seqs: np.ndarray of shape (N, SEQ_LEN)
    - meta_seqs: np.ndarray of shape (N, 3), where meta_seqs[i] = [patient_id, lead, index]
    """
    max_len = int(seq_len_seconds * fs)

    x_seqs = []
    y_seqs = []
    meta_seqs = []

    if beat_aligned:
        # Find p-wave start indices (transitions to label==1)
        is_p_wave = label_signal == 2
        transitions = np.diff(np.concatenate([[0], is_p_wave.astype(int)])) == 1
        p_wave_starts = np.where(transitions)[0]

        for i in range(len(p_wave_starts) - beats_per_seq):
            start = p_wave_starts[i]
            end = p_wave_starts[i + beats_per_seq]
            segment_len = end - start

            if segment_len > max_len:
                continue  # skip overly long sequences

            x_seg = ecg_signal[start:end]
            y_seg = label_signal[start:end]

            # Pad if too short
            if segment_len < max_len:
                pad_width = max_len - segment_len
                x_seg = np.pad(x_seg, (0, pad_width), constant_values=0.0)
                y_seg = np.pad(y_seg, (0, pad_width), constant_values=pad_label)

            patient_id = patient_signal[start]
            lead = lead_signal[start]
            meta_seqs.append([patient_id, lead, i])

            x_seqs.append(x_seg)
            y_seqs.append(y_seg)

    else:
        # Fixed-length segmentation
        total_len = len(ecg_signal)
        num_segments = total_len // max_len

        for i in range(num_segments):
            start = i * max_len
            end = start + max_len

            x_seg = ecg_signal[start:end]
            y_seg = label_signal[start:end]

            if len(x_seg) < max_len:
                pad_width = max_len - len(x_seg)
                x_seg = np.pad(x_seg, (0, pad_width), constant_values=0.0)
                y_seg = np.pad(y_seg, (0, pad_width), constant_values=pad_label)

            patient_id = patient_signal[start]
            lead = lead_signal[start]
            meta_seqs.append([patient_id, lead, i])

            x_seqs.append(x_seg)
            y_seqs.append(y_seg)

    return np.stack(x_seqs), np.stack(y_seqs), np.array(meta_seqs)

def select_random_sequences(x_seq, y_seq, n_samples, seed=None):
    """
    Randomly select a subset of sequences.

    Parameters:
    - x_seq: np.ndarray, shape (N, SEQ_LEN)
    - y_seq: np.ndarray, shape (N, SEQ_LEN)
    - n_samples: int, number of sequences to pick
    - seed: int or None, for reproducibility

    Returns:
    - x_sub: np.ndarray, shape (n_samples, SEQ_LEN)
    - y_sub: np.ndarray, shape (n_samples, SEQ_LEN)
    """
    if seed is not None:
        np.random.seed(seed)

    total_samples = x_seq.shape[0]
    if n_samples > total_samples:
        raise ValueError(f"Requested {n_samples} samples, but only {total_samples} available.")

    indices = np.random.choice(total_samples, size=n_samples, replace=False)

    x_sub = x_seq[indices]
    y_sub = y_seq[indices]

    return x_sub, y_sub

def extract_sequences_simple(
    ecg_signal, label_signal, fs,
    beats_per_seq=2, seq_len_seconds=2.0, pad_label=-100, beat_aligned=True
):
    max_len = int(seq_len_seconds * fs)

    x_seqs = []
    y_seqs = []

    if beat_aligned:
        is_p_wave = label_signal == 2
        transitions = np.diff(np.concatenate([[0], is_p_wave.astype(int)])) == 1
        p_wave_starts = np.where(transitions)[0]

        # Create start and end indices as arrays
        starts = p_wave_starts[:-beats_per_seq]
        ends = p_wave_starts[beats_per_seq:]

        segments = [(s, e) for s, e in zip(starts, ends) if (e - s) <= max_len and (e - s) >= 10]

        if not segments:
            raise ValueError("No valid beat-aligned segments found.")

        x_seqs = np.zeros((len(segments), max_len), dtype=ecg_signal.dtype)
        y_seqs = np.full((len(segments), max_len), pad_label, dtype=label_signal.dtype)

        for idx, (start, end) in enumerate(segments):
            seg_x = ecg_signal[start:end]
            seg_y = label_signal[start:end]

            x_seqs[idx, :len(seg_x)] = seg_x
            y_seqs[idx, :len(seg_y)] = seg_y

    else:
        # Fixed-length segmentation (already efficient)
        total_len = len(ecg_signal)
        num_segments = total_len // max_len

        if num_segments == 0:
            raise ValueError("ECG signal too short for even one sequence.")

        x_seqs = np.zeros((num_segments, max_len), dtype=ecg_signal.dtype)
        y_seqs = np.full((num_segments, max_len), pad_label, dtype=label_signal.dtype)

        for i in range(num_segments):
            start = i * max_len
            end = start + max_len
            x_seqs[i, :] = ecg_signal[start:end]
            y_seqs[i, :] = label_signal[start:end]

    return x_seqs, y_seqs

def get_exclude_sequence_ids(config):
    """
    Load merged sequence assessment CSV and return a list of sequence IDs to exclude.
    Parameters:
    - config: dict, should contain at least "base_path" where merged_seq_assessment.csv is stored
    Returns:
    - exclude_ids: list of integers, sequence IDs to exclude
    """
    # Path to merged CSV
    csv_path = f"{config['base_path']}/seq_assessment.csv"

    # Load CSV
    meta_seq = pd.read_csv(csv_path, keep_default_na=False)

    # Find sequences where Selected == "Yes"
    exclude_ids = meta_seq.loc[meta_seq["Selected"] == "Yes", "Sequence"].tolist()

    return exclude_ids

def load_ecg_data(config, split_ratio=1.0):
    """Load ECG signals, labels, patient IDs, and leads from a single CSV file, split by patient ID."""
    
    # Load the full CSV
    df = pd.read_csv(os.path.join(config["base_path"], "all_ecg_data.csv"), header=None)
    df.columns = ["ecg", "label", "patient_id", "lead"]
    
    # Get unique patient IDs
    patient_ids = df["patient_id"].unique()
    patient_ids.sort()
    
    # Split patients: 80% train, 20% test
    num_patients = len(patient_ids)
    num_train = int(split_ratio * num_patients)
    
    train_patients = patient_ids[:num_train]
    test_patients = patient_ids[num_train:]
    
    # Create train and test splits
    df_train = df[df["patient_id"].isin(train_patients)]
    df_test = df[df["patient_id"].isin(test_patients)]
    
    # Extract arrays
    x_train_raw = df_train["ecg"].values
    y_train_raw = df_train["label"].values
    patient_id_train = df_train["patient_id"].values
    lead_train = df_train["lead"].values
    
    x_test_raw = df_test["ecg"].values
    y_test_raw = df_test["label"].values
    patient_id_test = df_test["patient_id"].values
    lead_test = df_test["lead"].values
    
    return x_train_raw, y_train_raw, patient_id_train, lead_train, x_test_raw, y_test_raw, patient_id_test, lead_test

def split_train_val_by_patient(x_seq, y_seq, meta_seq, train_fraction=0.8, random_seed=42):
    """
    Splits ECG sequences into training and validation sets based on patient IDs to avoid data leakage.

    Parameters:
    - x_seq: np.ndarray, shape (N, sequence_length)
    - y_seq: np.ndarray, shape (N, sequence_length)
    - meta_seq: np.ndarray, shape (N, 2), where meta_seq[i] = [patient_id, lead]
    - train_fraction: float, fraction of patients to include in training set
    - random_seed: int, random seed for reproducibility

    Returns:
    - x_train: np.ndarray
    - y_train: np.ndarray
    - x_val: np.ndarray
    - y_val: np.ndarray
    """
    # Unique patients
    patient_ids = np.unique(meta_seq[:, 0])

    # Shuffle patient IDs
    rng = np.random.RandomState(random_seed)
    permuted_patients = rng.permutation(patient_ids)

    # Split patients
    n_train = int(train_fraction * len(permuted_patients))
    train_patients = permuted_patients[:n_train]
    val_patients = permuted_patients[n_train:]

    # Masks for selecting sequences
    train_mask = np.isin(meta_seq[:, 0], train_patients)
    val_mask = np.isin(meta_seq[:, 0], val_patients)

    # Split sequences
    x_train = x_seq[train_mask]
    y_train = y_seq[train_mask]
    x_val = x_seq[val_mask]
    y_val = y_seq[val_mask]

    return x_train, y_train, x_val, y_val

def filter_ecg(x, config):
    x_filt = apply_bandpass_filter(x, lowcut=0.5, highcut=40, fs=config["Fs"])
    return x_filt

def compute_class_weights(y_train_seq, num_classes):
    valid_classes = np.arange(num_classes)
    y_flat = y_train_seq.flatten()
    y_filtered = y_flat[y_flat != -100]
    return compute_class_weight(class_weight='balanced', classes=valid_classes, y=y_filtered)

def get_dataloaders(x_train_seq, y_train_seq, x_val_seq, y_val_seq, batch_size, num_workers=4):
    def to_torch(x, y):
        return (torch.tensor(x[:, :, np.newaxis], dtype=torch.float32), torch.tensor(y, dtype=torch.long))

    x_train_t, y_train_t = to_torch(x_train_seq, y_train_seq)
    x_val_t, y_val_t = to_torch(x_val_seq, y_val_seq)

    train_ds = TensorDataset(x_train_t, y_train_t)
    val_ds = TensorDataset(x_val_t, y_val_t)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    )

def get_device(config):
    device = config["device"]
    device_options = ["cuda", "cpu"]
    if device not in device_options:
        raise ValueError(f"valid device options are {device_options}")
    if device == "cuda" and not torch.cuda.is_available():
        ValueError("cuda is not available")
    return device

def build_model(config):
    with redirect_output_to_file(): # Suppress output from the model initialization as it can be verbose
        return get_model(config["model"]["name"], **config["model"]["params"])
    
def train_epoch(model, loader, optimizer, loss_fn, device=None):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    if device == None:
        device = next(model.parameters()).device

    num_batches = len(loader)
    tqdm_loader = tqdm(enumerate(loader), total=num_batches)
    for batch_idx, (x, y) in tqdm_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)

        # Optional: Clamp logits to avoid overflow (can still keep this)
        #logits = torch.clamp(logits, min=-15, max=15)

        # Cross-entropy loss
        loss_ce = loss_fn(logits.permute(0, 2, 1), y).mean()

        # Temporal consistency
        # valid_mask = (y != -100).float()
        # valid_mask_diff = valid_mask[:, 1:] * valid_mask[:, :-1]

        # probs = torch.softmax(logits, dim=-1)
        # diff = probs[:, 1:, :] - probs[:, :-1, :]
        # denom = valid_mask_diff.sum() + 1e-8
        # temporal_consistency = (diff.abs().sum(-1) * valid_mask_diff).sum() / denom

        # pred_classes = torch.argmax(probs, dim=-1)  # (B, T)
        # label_switches = (pred_classes[:, 1:] != pred_classes[:, :-1]).float()
        # label_switches = label_switches * valid_mask_diff  # mask invalid transitions

        # change_loss = label_switches.sum() / (valid_mask_diff.sum() + 1e-8)

        loss = loss_ce

        tqdm_loader.set_postfix(loss=f"{loss.item():.3f}")
        tqdm_loader.update()

        if torch.isnan(loss):
            total_loss = torch.nan
            break

        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item()
    tqdm_loader.close()
    duration = time.time() - start_time
    print(f"🕒 Train epoch completed in {duration:.2f} seconds")
    return total_loss / len(loader)

def validate(model, loader, loss_fn, device=None):
    model.eval()
    total_loss = 0.0
    start_time = time.time()

    if device == None:
        device = next(model.parameters()).device

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.permute(0, 2, 1), y)
            total_loss += loss.mean().item()

    duration = time.time() - start_time
    print(f"🕒 Validation completed in {duration:.2f} seconds")
    return total_loss / len(loader)

def write_hparams(writer, config, val_loss=0.0, macrof1=0.0):
    # checm if embedding_dim, conv1d_kernel_size, num_blocks, dropout, num_heads, slstm_at are in config; if not, put "n/a" in config
    # for key in ['embedding_dim', 'conv1d_kernel_size', 'num_blocks', 'dropout', 'num_heads', 'slstm_at']:
    #     if key not in config:
    #         config[key] = "n/a"

    hparams = {
        **config["model"]["params"],
        'seq_len': config['SEQ_LEN'],
        'batch_size': config['batch_size'],
        'initial_lr': config['initial_lr'],
    }
    hparams = {
        k: str(v) for k, v in hparams.items()
    }
    # hparams = {
    #     'embedding_dim': config['embedding_dim'],
    #     'kernel_size': config['conv1d_kernel_size'],
    #     'num_blocks': config['num_blocks'],
    #     'dropout': config['dropout'],
    #     'num_heads': config['num_heads'],
    #     'slstm_at': str(config['slstm_at']),
    #     'seq_len': config['SEQ_LEN'],
    #     'batch_size': config['batch_size'],
    #     'initial_lr': config['initial_lr'],
    # }
    metrics = {
        'val_loss': float(val_loss),        
        'val_f1_macro': float(macrof1),
    }
    writer.add_hparams(hparams, metrics)

def plot_sequence_shaded(ecg_signal, true_labels, raw_preds, SEQ_LEN, debug=False):
    """
    Plots ECG signal with two ribbon overlays:
    - Ground truth labels (top ribbon, lighter flare colormap)
    - Predicted labels (bottom ribbon, normal flare colormap)
    No full background shading.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.colors import to_rgba

    Fs = 250
    t = np.arange(SEQ_LEN) / Fs

    ecg_signal = np.asarray(ecg_signal)
    true_labels = np.asarray(true_labels)
    raw_preds = np.asarray(raw_preds)

    sns.set_style("ticks")  # <-- no grid, clean white background
    sns.set_context("notebook", rc={
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot ECG signal
    ax.plot(t, ecg_signal, color='#444444', linewidth=2.0, label="ECG", zorder=2)

    # Define ribbons: truth and prediction
    ribbon_gap = 0.05 * (np.max(ecg_signal) - np.min(ecg_signal))  # Gap between ribbons
    ribbon_height = 0.05 * (np.max(ecg_signal) - np.min(ecg_signal))  # Height of ribbons

    # Ribbon positions
    truth_bottom = np.min(ecg_signal) - 2 * ribbon_gap - 2 * ribbon_height
    truth_top = np.min(ecg_signal) - 1 * ribbon_gap - 1 * ribbon_height
    pred_bottom = np.min(ecg_signal) - 1 * ribbon_gap - 1 * ribbon_height
    pred_top = np.min(ecg_signal) - 0 * ribbon_gap

    # Use flare colormap for both
    cmap_flare = sns.color_palette("flare", as_cmap=True)

    n_classes = max(np.max(true_labels), np.max(raw_preds)) + 1 if max(np.max(true_labels), np.max(raw_preds)) > 0 else 1

    # Helper to lighten a color
    def lighten_color(color, factor=0.5):
        r, g, b, a = to_rgba(color)
        return (r + (1 - r) * factor,
                g + (1 - g) * factor,
                b + (1 - b) * factor,
                a)

    # Plot ground truth ribbon (lighter)
    for i in range(len(t) - 1):
        label = true_labels[i]
        if label in [-100, 0] or np.isnan(label):
            continue
        color = cmap_flare(label / max(n_classes-1, 1))
        color_light = lighten_color(color, factor=0.5)  # 50% lighter
        ax.fill_between(
            t[i:i+2],
            truth_bottom, truth_top,
            color=color_light,
            edgecolor=None,
            linewidth=0,
            zorder=0
        )

    # Plot prediction ribbon (normal)
    for i in range(len(t) - 1):
        label = raw_preds[i]
        if label in [-100, 0] or np.isnan(label):
            continue
        color = cmap_flare(label / max(n_classes-1, 1))
        ax.fill_between(
            t[i:i+2],
            pred_bottom, pred_top,
            color=color,
            edgecolor=None,
            linewidth=0,
            zorder=1
        )

    # Labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (n.u.)")
    ax.grid(False)
    fig.tight_layout()
    sns.despine()

    return fig

def apply_classwise_tolerance_matching(y_true_seq, y_pred_seq, fs, tolerance_map, ignore_index=-100):
    """
    Correct predictions within a class-specific temporal tolerance.
    """
    y_pred_tolerant = np.copy(y_pred_seq)
    for cls, tol_ms in tolerance_map.items():
        tol = int((tol_ms / 1000) * fs)
        for i in range(y_true_seq.shape[0]):
            for t in range(y_true_seq.shape[1]):
                if y_true_seq[i, t] != cls:
                    continue
                if y_true_seq[i, t] == ignore_index:
                    continue
                start = max(0, t - tol)
                end = min(y_pred_seq.shape[1], t + tol + 1)
                if cls in y_pred_seq[i, start:end]:
                    y_pred_tolerant[i, t] = cls
    return y_pred_tolerant

def fig_to_tensorboard_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    return ToTensor()(image)

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
        smoothed[:, i] = scipy.stats.mode(window, axis=1, keepdims=False)[0]

    return smoothed

def save_model_with_metadata(model, config, results_path, best_macro_f1, epoch,
                             train_loss, val_loss, current_lr):
    """
    Save model checkpoint and metadata to a timestamped directory.

    Arguments:
        model: PyTorch model
        config: full config dict (must contain 'name')
        results_path: relative or absolute path (e.g., "res" or "/path/to/res")
        best_macro_f1: best macro-F1 score so far
        epoch: current epoch
        train_loss, val_loss, current_lr: training stats

    Returns:
        model_filename (str): Absolute path to the saved model file.
    """
    # Resolve absolute base path regardless of script location
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    abs_results_path = os.path.join(base_dir, results_path)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(abs_results_path, f"{config['name']}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    model_filename = os.path.join(model_dir, f"macroF1_{best_macro_f1:.4f}_epoch{epoch+1}_{timestamp}.pt")
    torch.save(model.state_dict(), model_filename)

    metadata = {
        "epoch": epoch + 1,
        "macro_f1": float(best_macro_f1),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "learning_rate": float(current_lr),
        "timestamp": timestamp,
        "config": config
    }

    metadata_filename = model_filename.replace(".pt", ".json")
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved new best model: {model_filename} and metadata")
    return model_filename

def extract_all_ecg_features(preds, ecg_signal, Fs=250):
    """
    Extract ECG morphology and HRV features from predicted delineation and ECG signal.
    
    Args:
        preds: 1D numpy array of predicted labels (0=bg, 1=P, 2=QRS, 3=T)
        ecg_signal: 1D numpy array of raw ECG values (same length as preds)
        Fs: Sampling frequency in Hz

    Returns:
        Dictionary with extracted ECG features
    """
    ms_per_sample = 1000 / Fs
    waves = {1: 'P', 2: 'QRS', 3: 'T'}
    boundaries = {'P': [], 'QRS': [], 'T': []}
    current_label = None
    start_idx = None

    for i, label in enumerate(preds):
        if label != current_label:
            if current_label in waves and start_idx is not None:
                boundaries[waves[current_label]].append((start_idx, i - 1))
            if label in waves:
                start_idx = i
            else:
                start_idx = None
            current_label = label
    if current_label in waves and start_idx is not None:
        boundaries[waves[current_label]].append((start_idx, len(preds) - 1))

    features = {
        'PR_intervals': [], 'QRS_durations': [], 'QT_intervals': [],
        'P_durations': [], 'T_durations': [], 'ST_segments': [],
        'P_amplitudes': [], 'QRS_amplitudes': [], 'T_amplitudes': [],
        'QRS_to_P_amp_ratios': [],
        'RR_intervals': [], 'HR': [], 'SDNN': [], 'RMSSD': [], 'NN50': [], 'pNN50': []
    }

    qrs_onsets = [start for start, _ in boundaries['QRS']]

    for p_start, p_end in boundaries['P']:
        qrs_after_p = [q_start for q_start in qrs_onsets if q_start > p_end]
        if qrs_after_p:
            pr_interval = (qrs_after_p[0] - p_start) * ms_per_sample
            features['PR_intervals'].append(pr_interval)

    for q_start, q_end in boundaries['QRS']:
        duration = (q_end - q_start + 1) * ms_per_sample
        features['QRS_durations'].append(duration)

        t_candidates = [t for t in boundaries['T'] if t[0] > q_end]
        if t_candidates:
            t_start, t_end = t_candidates[0]
            qt_interval = (t_end - q_start + 1) * ms_per_sample
            features['QT_intervals'].append(qt_interval)
            st_segment = (t_start - q_end - 1) * ms_per_sample
            features['ST_segments'].append(st_segment)

    for p_start, p_end in boundaries['P']:
        duration = (p_end - p_start + 1) * ms_per_sample
        features['P_durations'].append(duration)

    for t_start, t_end in boundaries['T']:
        duration = (t_end - t_start + 1) * ms_per_sample
        features['T_durations'].append(duration)

    for p_start, p_end in boundaries['P']:
        segment = ecg_signal[p_start:p_end + 1]
        if len(segment) > 0:
            amp = np.max(segment) - np.min(segment)
            features['P_amplitudes'].append(amp)

    for q_start, q_end in boundaries['QRS']:
        segment = ecg_signal[q_start:q_end + 1]
        if len(segment) > 0:
            amp = np.max(segment) - np.min(segment)
            features['QRS_amplitudes'].append(amp)

    for t_start, t_end in boundaries['T']:
        segment = ecg_signal[t_start:t_end + 1]
        if len(segment) > 0:
            amp = np.max(segment) - np.min(segment)
            features['T_amplitudes'].append(amp)

    min_len = min(len(features['P_amplitudes']), len(features['QRS_amplitudes']))
    for i in range(min_len):
        p_amp = features['P_amplitudes'][i]
        qrs_amp = features['QRS_amplitudes'][i]
        if p_amp > 0:
            ratio = qrs_amp / p_amp
            if 0 < ratio < 50:
                features['QRS_to_P_amp_ratios'].append(ratio)

    # --- HRV features ---
    if len(qrs_onsets) >= 2:
        rr_intervals = np.diff(qrs_onsets) * ms_per_sample  # in ms
        features['RR_intervals'].extend(rr_intervals.tolist())

        if len(rr_intervals) > 0:
            features['HR'].append(60000 / np.mean(rr_intervals))  # bpm
            features['SDNN'].append(np.std(rr_intervals, ddof=1))  # ms

        if len(rr_intervals) > 1:
            successive_diff = np.diff(rr_intervals)
            features['RMSSD'].append(np.sqrt(np.mean(successive_diff ** 2)))  # ms

            nn50 = np.sum(np.abs(successive_diff) > 50)
            pnn50 = 100 * nn50 / len(successive_diff)
            features['NN50'].append(nn50)
            features['pNN50'].append(pnn50)

    return features

def build_feature_pipeline(feature_df):
    # ---------- Feature columns ----------
    feature_cols = [col for col in feature_df.columns if col.startswith("mean_")]
    
    # ---------- Config ----------
    skewed_features = ['QRS_to_P_amp_ratios']  # Add more if needed
    log_cols = [col for col in feature_cols if any(s in col for s in skewed_features)]
    nonlog_cols = [col for col in feature_cols if col not in log_cols]

    # ---------- Winsorization ----------
    def winsorize_column(col):
        return winsorize(col, limits=[0.01, 0.01])

    def apply_winsorization(X):
        X = np.asarray(X)  # ✅ This fixes the KeyError
        return np.column_stack([winsorize_column(X[:, i]) for i in range(X.shape[1])])

    def log_transform(X):
        return np.log1p(X)

    # ---------- Pipelines ----------
    nonlog_pipeline = Pipeline([
        ("winsorize", FunctionTransformer(apply_winsorization, validate=False)),
        ("scale", RobustScaler())
    ])

    log_pipeline = Pipeline([
        ("winsorize", FunctionTransformer(apply_winsorization, validate=False)),
        ("log", FunctionTransformer(log_transform, validate=False)),
        ("scale", RobustScaler())
    ])

    # ---------- Combine ----------
    full_pipeline = ColumnTransformer([
        ("log", log_pipeline, log_cols),
        ("nonlog", nonlog_pipeline, nonlog_cols)
    ])

    return full_pipeline, feature_cols

def load_ptbxl_data(
    path="../DATA/ptb-xl",
    sampling_rate=500,
    target_fs=250,
    leads='all',  # 'all' or list of indices, e.g., [1, 5]
    use_superclass=True
):
    """
    Load PTB-XL ECG data with labels.

    Args:
        path: Path to PTB-XL dataset directory (must include CSVs and WFDB files)
        sampling_rate: 100 or 500 (must match subfolder structure)
        target_fs: resample signals to this (e.g., 250)
        leads: 'all' or list of indices to select from 12 leads
        use_superclass: if True, assign diagnostic_superclass from SCP codes

    Returns:
        X: np.array of shape (N, L, T) if multilead, or (N, T) if single-lead
        y: list of labels (multi-label format)
        metadata: corresponding dataframe
    """
    print(f"📦 Loading PTB-XL at {sampling_rate} Hz...")

    # Load metadata
    df = pd.read_csv(f"{path}/ptbxl_database.csv", index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    # Load SCP code aggregation
    agg_df = pd.read_csv(f"{path}/scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        return list({agg_df.loc[k].diagnostic_class for k in y_dic if k in agg_df.index})

    if use_superclass:
        df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
        y = df.diagnostic_superclass
    else:
        y = df.scp_codes

    # Load raw ECGs
    file_col = "filename_hr" if sampling_rate == 500 else "filename_lr"
    records = df[file_col].values
    x_list = []

    for f in records:
        record = wfdb.rdsamp(f"{path}/{f}")
        signal = record[0].T  # shape (12, T)
        if sampling_rate != target_fs:
            signal = resample(signal, int(signal.shape[1] * target_fs / sampling_rate), axis=1)
        if leads != 'all':
            signal = signal[leads, :]
        x_list.append(signal)

    X = np.stack(x_list)  # (N, L, T) or (N, T) if L = 1
    y = y.tolist()

    print(f"✅ Loaded {len(X)} samples at {target_fs} Hz with {X.shape[1]} leads each.")
    return X, y, df
