import os
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset


from helper_fns_new import (
    extract_sequences_simple, z_normalize,
    select_random_sequences
)

from src.utils.validation_helper_fns import (
    apply_bandpass_filter, print_ascii_logo, 
    evluate_on_db, load_model, get_tolerance_map
)

def main(model_name, data_dir="../DATA/ludb", filter_data=True, beat_aligned=True, use_subset_for_testing=False, n_sub_samples=1000, seed=42, results_dir="./results", db_name="ludb"):
    print("✨ Start processing...")

    # Load model
    print("🔍 Loading model...")
    model, config, Fs, model_path = load_model(model_name)

    # Load data
    try:
        # Try loading .npz format
        x_data = np.load(os.path.join(data_dir, f"fs{Fs}_x.npz"))["arr_0"]
        y_data = np.load(os.path.join(data_dir, f"fs{Fs}_y.npz"))["arr_0"]
        print("✅ Loaded data from .npz files.")
    except (FileNotFoundError, KeyError):
        # Fallback: try loading .npy format instead
        print("⚠️ .npz files not found or invalid. Trying to load .npy files instead...")
        x_data = np.load(os.path.join(data_dir, f"fs{Fs}_x.npy"))
        y_data = np.load(os.path.join(data_dir, f"fs{Fs}_y.npy"))
        print("✅ Loaded data from .npy files.")

    # Preprocessing
    print(f"🔄 Extracting sequences ({config['heart_beats']} heart beats, {config['seq_dur']} seconds), beat aligned extraction: {beat_aligned}...")
    x_seq, y_seq = extract_sequences_simple(
        ecg_signal=x_data,
        label_signal=y_data,
        fs=Fs,
        beats_per_seq=config.get('heart_beats', 1),
        seq_len_seconds=config.get('seq_dur', 1.2),
        pad_label=-100, beat_aligned=True
    )
    
    # Apply bandpass filter
    if filter_data:
        print("🧹 Applying bandpass filtering (0.5–50 Hz)...")
        #x_data = apply_bandpass_filter(x_data, fs=Fs)
        x_seq = np.array([apply_bandpass_filter(seq, fs=Fs) for seq in x_seq])

    # Normalize
    print("🔄 Normalizing sequences...")
    x_seq = np.array([z_normalize(seq) for seq in x_seq])
    
    if use_subset_for_testing:
        print(f"🔄 Subsampling to {n_sub_samples} sequences...")
        x_seq, y_seq = select_random_sequences(x_seq, y_seq, n_samples=n_sub_samples, seed=seed)

    # Torch tensors
    x = torch.tensor(x_seq[:, :, np.newaxis], dtype=torch.float32)
    y = torch.tensor(y_seq, dtype=torch.long)

    dataset = TensorDataset(x, y)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ========== Full Model Evaluation ==========
    save_dir_full = f"{results_dir}/{model_name}/{db_name}"
    full_info, plots_data = evluate_on_db(
        model=model,
        test_loader=loader,
        x_test_tensor=x,
        y_test_tensor=y,
        SEQ_LEN=config['SEQ_LEN'],
        save_dir=save_dir_full,
        Fs=Fs,
        target_names=config['target_names'],
        tolerance_map=get_tolerance_map(),
        loader=loader,
        model_path=model_path,
    )

    with open(os.path.join(save_dir_full, "model_info.json"), "w") as f:
        json.dump(full_info, f, indent=4)

    # Save all collected plot data
    with open(os.path.join(save_dir_full, "plots_data.pkl"), "wb") as f:
        pickle.dump(plots_data, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="🔥 BixECG Evaluation Script: Evaluate an ECG model on LUDB with full reporting, calibration, uncertainty, tolerance-aware evaluation, and shaded sequence plots."
    )

    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the saved model checkpoint (without extension).")
    parser.add_argument("--data_dir", type=str, default="../DATA/ludb",
                        help="Path to the ECG database directory.")
    parser.add_argument("--filter_data", action="store_true",
                        help="Apply preprocessing filters (e.g., bandpass) to the ECG data before evaluation.")
    parser.add_argument("--beat_aligned", action="store_true",
                        help="Use beat-aligned ECG segmentation instead of fixed-size windows.")
    parser.add_argument("--use_subset_for_testing", action="store_true",
                        help="Use only a subset of the dataset for testing (useful for quick runs).")
    parser.add_argument("--n_sub_samples", type=int, default=1000,
                        help="Number of samples to use when --use_subset_for_testing is enabled.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory where evaluation results (plots, metrics, reports) will be saved.")
    parser.add_argument("--db_name", type=str, default="ludb",
                        help="Name of the database for labeling purposes in results.")

    return parser.parse_args()

def main_cli():
    print_ascii_logo()
    args = parse_args()

    main(
        model_name=args.model_name,
        data_dir=args.data_dir,
        filter_data=args.filter_data,
        beat_aligned=args.beat_aligned,
        use_subset_for_testing=args.use_subset_for_testing,
        n_sub_samples=args.n_sub_samples,
        seed=args.seed,
        results_dir=args.results_dir,
        db_name=args.db_name
    )
    print("\n✅ All tasks completed successfully!")

if __name__ == "__main__":
    # call e.g. with best_model_macroF1_0.9219_epoch3_20250420_192953
    main_cli()
