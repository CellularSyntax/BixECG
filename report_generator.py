import os
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool

def read_json(file):
    with open(file) as f:
        d = json.load(f)
    return d

def flattenjson(b, delim):
    val = {}
    for i in b.keys():
        if isinstance(b[i], dict):
            get = flattenjson(b[i], delim)
            for j in get.keys():
                val[i + delim + j] = get[j]
        else:
            val[i] = b[i]
            
    return val

def run_validate_script(x):
    model_name, beat_aligned, report_only = x
    cmd = f"python validate_model.py --model_name {model_name}"
    if not beat_aligned:
        cmd += " --no-beat_aligned"
    if report_only:
        cmd += " --report_only"
    print(f"Running command: {cmd}")
    os.system(cmd)


def validate(root_dir, beat_aligned=True, pool_size=7, report_only=True):
    # python validate_model.py --no-beat_aligned --report_only --model_name {root_dir}/_test_bs256_seqlen300_emb8_ks51_blocks3_nh2_slstmat\[\]_do0.2_lr0.001_20250512_204037/macroF1_0.9046_epoch19.pt

    p = Pool(pool_size)

    model_names = []

    summary_csv = os.path.join(root_dir, "summary.csv")
    df = pd.read_csv(summary_csv, delimiter=";")
    # iterate over all rows of the dataframe
    for index, row in df.iterrows():
        macrof1_training = row["best_macrof1"]
        
        if macrof1_training == 0:
            continue

        config = row["config"]

        run_dir = os.path.join(root_dir, config)

        # find all files ending with .pt in the run_dir
        files = sorted([f for f in os.listdir(run_dir) if f.endswith(".pt")])
        best_model = files[-1]

        model = os.path.join(run_dir, best_model)

        model_names.append([f"{model}", beat_aligned, report_only])


    p.map(run_validate_script, model_names)


def merge_reports(root_dir, beat_aligned=True):
    print("✨ Start processing...")

    
    summary_csv = os.path.join(root_dir, "summary.csv")

    

    report_runs = []

    # load content of summary.csv into a pandas dataframe
    df = pd.read_csv(summary_csv, delimiter=";")
    # iterate over all rows of the dataframe
    for index, row in df.iterrows():
        report = {}
        macrof1_training = row["best_macrof1"]
        
        if macrof1_training == 0:
            continue
        
        config = row["config"]
        num_params = row["total_params"]

        report["config"] = config
        report["total_params"] = num_params
        report["macrof1_training"] = macrof1_training
        report["stopped_due_to"] = row["stopped_due_to"]

        # get subdirectory with name starting with config
        subdir = [d for d in os.listdir(root_dir) if d.startswith(config)][0]
        print(f"Subdir: {subdir}")
        subdir = os.path.join(root_dir, subdir)


        # iterate over all files recursively inside model_info_json_path
        for root, dirs, files in os.walk(subdir):
            for file in files:
                file_path = os.path.join(root, file)
                if beat_aligned and "aligned" not in file_path:
                    continue
                if not beat_aligned and "fixed/model_info" not in file_path:
                    continue
                if file.endswith("model_info.json"):
                    
                    print(f"Reading {file_path}")
                    
                    # read json file
                    data = read_json(file_path)
                    # flatten json
                    data = flattenjson(data, "__")
                    # add to report
                    report.update(data)
        report_runs.append(report)

    alignment = "aligned" if beat_aligned else "fixed"
    merged_csv_path = os.path.join(root_dir, f"merged_{alignment}.csv")
    # Remove summary.csv if it exists
    if os.path.exists(merged_csv_path):
        os.remove(merged_csv_path)
        print(f"✅ Removed existing merged.csv file.")
    # Create merged.csv file without writing anything to it
    with open(merged_csv_path, "w") as f:
        for (i, run_report) in enumerate(report_runs):
            if i == 0:
                # write header
                f.write(";".join(run_report.keys()) + "\n")
            f.write(";".join([str(run_report[k]) for k in run_report.keys()]) + "\n")

    print(f"✅ Merged {len(report_runs)} runs into {merged_csv_path} file.")


    



if __name__ == "__main__":

    root_dir = "results/BixLSTM_nonshared_aligned"
    validate(root_dir, beat_aligned=True, report_only=True)
    validate(root_dir, beat_aligned=False, report_only=True)

    merge_reports(root_dir, beat_aligned=True)
    merge_reports(root_dir, beat_aligned=False)

