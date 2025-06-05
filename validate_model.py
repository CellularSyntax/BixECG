import os
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

import pickle


from src.utils.helper_fns import (
    extract_sequences_simple, z_normalize,
    select_random_sequences,
    extract_sequences_starting_with_wave
)

from src.utils.validation_helper_fns import (
    apply_bandpass_filter, print_ascii_logo, 
    evluate_on_db, load_model, get_tolerance_map
)

def main(model_name, 
         data_dir="../DATA/ludb", 
         filter_data=True, 
         beat_aligned=True, 
         pad=True,
         use_subset_for_testing=False, 
         n_sub_samples=1000, 
         seed=42, 
         results_dir="./results", 
         db_name="ludb", 
         report_only=False, 
         batch_size=None
):
    print("✨ Start processing...")
    print("beat_aligned " + str(beat_aligned))
    print("pad " + str(pad))
    print("filter_data " + str(filter_data))

    # Load model
    print("🔍 Loading model...")
    model, config, Fs, model_path = load_model(model_name)

    if batch_size is None:
        batch_size = config["batch_size"]


    if db_name == "qtdb":
        # Load the full CSV
        df = pd.read_csv(os.path.join(config["base_path"], "all_ecg_data.csv"), header=None)
        df.columns = ["ecg", "label", "patient_id", "lead"]
        
        x_data = df["ecg"].values
        y_data = df["label"].values

    elif db_name == "ludb_qrs_aligned_seq":
        with open(os.path.join(data_dir, f"fs{Fs}_x_qrs_aligned_seq.pickle"), 'rb') as handle:
            x_data = pickle.load(handle)
        with open(os.path.join(data_dir, f"fs{Fs}_y_qrs_aligned_seq.pickle"), 'rb') as handle:
            y_data = pickle.load(handle)

        x_seq = []
        y_seq = []
        for rec in x_data.keys():
            for lead in x_data[rec].keys():
                for i in range(len(x_data[rec][lead])):
                    i_seq_x = x_data[rec][lead][i]
                    i_seq_y = y_data[rec][lead][i]

                    if pad:
                        is_qrs = i_seq_y == 2
                        transitions = np.diff(np.concatenate([[0], is_qrs.astype(int)])) == 1
                        wave_starts = np.where(transitions)[0]
                        if len(wave_starts) > 1:
                            i_seq_x[wave_starts[1]:] = 0
                            i_seq_y[wave_starts[1]:] = 0
                        

                    x_seq.append(i_seq_x)
                    y_seq.append(i_seq_y)

    elif db_name == "ludb_qrs_aligned_padded" or db_name.startswith("ludb_lead") or db_name == "lvad_ecgs_new_qrs_aligned_padded" or db_name == "rabbit_qrs_aligned_padded":
        with open(os.path.join(data_dir, f"fs{Fs}_x.pickle"), 'rb') as handle:
            x_data = pickle.load(handle)
        with open(os.path.join(data_dir, f"fs{Fs}_y.pickle"), 'rb') as handle:
            y_data = pickle.load(handle)
        x_seq = x_data
        y_seq = y_data
        print(f"Loaded {len(x_seq)} sequences from {db_name}.")
    
        # x_seq = []
        # y_seq = []
        # for rec in x_data.keys():
        #     for lead in x_data[rec].keys():
        #         for i in range(len(x_data[rec][lead])):
        #             x_seq.append(x_data[rec][lead][i])
        #             y_seq.append(y_data[rec][lead][i])

    else:
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

        # x_seq, y_seq = extract_sequences_starting_with_wave(
        #     ecg_signal=x_data,
        #     label_signal=y_data,
        #     fs=Fs,
        #     seq_len_seconds=config.get('seq_dur', 1.2),
        # )

        x_seq, y_seq = extract_sequences_simple(
            ecg_signal=x_data,
            label_signal=y_data,
            beats_per_seq=config.get('heart_beats', 1),
            fs=Fs,
            seq_len_seconds=config.get('seq_dur', 1.2),
        )
    # if db_name != "ludb_qrs_aligned_seq":
    #     # Preprocessing
    #     print(f"🔄 Extracting sequences ({config['heart_beats']} heart beats, {config['seq_dur']} seconds), beat aligned extraction: {beat_aligned}...")
    #     x_seq, y_seq = extract_sequences_simple(
    #         ecg_signal=x_data,
    #         label_signal=y_data,
    #         fs=Fs,
    #         beats_per_seq=config.get('heart_beats', 1),
    #         seq_len_seconds=config.get('seq_dur', 1.2),
    #         pad_label=-100, beat_aligned=beat_aligned,
    #         pad=pad
    #     )

    
    
    # Apply bandpass filter
    
        #x_data = apply_bandpass_filter(x_data, fs=Fs)
    _x_seq = []
    for i in range(len(x_seq)):
        if filter_data and i == 0:
            print("🧹 Applying bandpass filtering (0.5–50 Hz)...")
        # find index of first element with value 2
        seq = x_seq[i]
        if np.any(y_seq[i] == -100):
            first_pad_index = np.where(y_seq[i] == -100)[0][0]
            #print(f"Found pad at index {first_pad_index} in sequence {i}. length x: {len(x_seq)} y: {len(y_seq)}")
            # Apply filter only to the part of the sequence before the first QRS
            if filter_data:
                seq[:first_pad_index] = z_normalize(apply_bandpass_filter(seq[:first_pad_index], fs=Fs))
            else:
                seq[:first_pad_index] = z_normalize(seq[:first_pad_index])
        else:
            # If no pad is present, apply filter to the whole sequence
            if filter_data:
                seq = apply_bandpass_filter(x_seq[i], fs=Fs)
            else:
                seq = z_normalize(x_seq[i])
        _x_seq.append(seq)
    x_seq = np.array(_x_seq)

    # Normalize
    #print("🔄 Normalizing sequences...")
    #x_seq = np.array([z_normalize(seq) for seq in x_seq])

    # for i in range(len(x_seq)):
    #     # find index of first element with value 2
    #     seq = x_seq[i]
    #     if np.any(y_seq[i] == -100):
    #         first_pad_index = np.where(y_seq[i] == -100)[0][0]
    #         # Apply filter only to the part of the sequence before the first QRS
    #         seq[:first_pad_index] = z_normalize(seq[:first_pad_index])
    #     else:
    #         # If no pad is present, apply filter to the whole sequence
    #         seq = z_normalize(seq[i])
    #     x_seq[i] = seq
    # print(f"normalized min = {np.min(x_seq)}, max = {np.max(x_seq)}")
    
    if use_subset_for_testing:
        print(f"🔄 Subsampling to {n_sub_samples} sequences...")
        x_seq, y_seq = select_random_sequences(x_seq, y_seq, n_samples=n_sub_samples, seed=seed)

    # Torch tensors
    x = torch.tensor(x_seq[:, :, np.newaxis], dtype=torch.float32)
    y = torch.tensor(y_seq, dtype=torch.long)

    dataset = TensorDataset(x, y)
    loader = DataLoader(
        dataset,
        #batch_size=config["batch_size"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ========== Full Model Evaluation ==========
    aligned = "aligned" if beat_aligned else "fixed"
    save_dir_full = os.path.join(model_name, db_name, aligned)

    if report_only:
        full_info = evluate_on_db(
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
            report_only=True,
        )
    else:
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
            report_only=False,
        )
        # Save all collected plot data
        with open(os.path.join(save_dir_full, "plots_data.pkl"), "wb") as f:
            pickle.dump(plots_data, f)

    with open(os.path.join(save_dir_full, "model_info.json"), "w") as f:
        json.dump(full_info, f, indent=4)

    


def parse_args():
    parser = argparse.ArgumentParser(
        description="🔥 BixECG Evaluation Script: Evaluate an ECG model on LUDB with full reporting, calibration, uncertainty, tolerance-aware evaluation, and shaded sequence plots."
    )

    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the saved model checkpoint.")
    parser.add_argument("--data_dir", type=str, default="../DATA/ludb",
                        help="Path to the ECG database directory.")
    parser.add_argument("--filter_data", action="store_true",
                        help="Apply preprocessing filters (e.g., bandpass) to the ECG data before evaluation.")
    parser.add_argument("--beat_aligned", dest="beat_aligned", default=True, action="store_true",
                        help="Use beat-aligned ECG segmentation instead of fixed-size windows.")
    parser.add_argument("--no-beat_aligned", dest="beat_aligned", action="store_false",
                        help="")
    parser.add_argument("--pad", dest="pad", default=True, action="store_true",
                        help="")
    parser.add_argument("--no-pad", dest="pad", action="store_false",
                        help="")
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
    parser.add_argument('--report_only', default=False, action=argparse.BooleanOptionalAction,
                        help="Create report only without creating plots. Default: False")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for DataLoader. If not specified, uses the default from the model config.")
    

    return parser.parse_args()

def main_cli():
    print_ascii_logo()
    args = parse_args()
    model_name = args.model_name
    if model_name.endswith(".pt") or model_name.endswith(".pth"):
        model_name = os.path.splitext(args.model_name)[0]

    main(
        model_name=model_name,
        data_dir=args.data_dir,
        filter_data=args.filter_data,
        beat_aligned=args.beat_aligned,
        pad=args.pad,
        use_subset_for_testing=args.use_subset_for_testing,
        n_sub_samples=args.n_sub_samples,
        seed=args.seed,
        results_dir=args.results_dir,
        db_name=args.db_name,
        report_only=args.report_only,
        batch_size=args.batch_size
    )
    print("\n✅ All tasks completed successfully!")

if __name__ == "__main__":
    # call e.g. with best_model_macroF1_0.9219_epoch3_20250420_192953
    main_cli()
