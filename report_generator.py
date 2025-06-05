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
    model_name, index, beat_aligned, pad, report_only, data_dir, db_name, batch_size, filter_data = x
    cmd = f"python validate_model.py --model_name {model_name} --data_dir ../DATA/{data_dir} --db_name {db_name} --batch_size {batch_size}"
    if not beat_aligned:
        cmd += " --no-beat_aligned"
    if report_only:
        cmd += " --report_only"
    if filter_data:
        cmd += " --filter_data"
    if pad:
        cmd += " --pad"
    else:
        cmd += " --no-pad"
    print(f"Running {index}. command: {cmd}")
    os.system(cmd)


def validate(root_dir, beat_aligned=True, pad=True, pool_size=5, report_only=True, data_dir="ludb", db_name="ludb", batch_size=64, filter_data=False):
    # python validate_model.py --no-beat_aligned --report_only --model_name {root_dir}/_test_bs256_seqlen300_emb8_ks51_blocks3_nh2_slstmat\[\]_do0.2_lr0.001_20250512_204037/macroF1_0.9046_epoch19.pt

    p = Pool(pool_size)

    model_names = []

    summary_csv = os.path.join(root_dir, "summary.csv")
    

    with open(summary_csv) as f:
        for (i, line) in enumerate(f):
            if i == 0:
                continue
            macrof1_training, config = line.split(";")[0:2]
            macrof1_training = float(macrof1_training)
            if macrof1_training == 0:
                continue
            run_dir = os.path.join(root_dir, config)
            # find all files ending with .pt in the run_dir
            files = sorted([f for f in os.listdir(run_dir) if f.endswith(".pt")])
            best_model = files[-1]

            model = os.path.join(run_dir, best_model)

            model_names.append([f"{model}", i+1, beat_aligned, pad, report_only, data_dir, db_name, batch_size, filter_data])


    # df = pd.read_csv(summary_csv, delimiter=";")
    



    # # iterate over all rows of the dataframe in order
    # for index, row in df.iterrows():
    #     macrof1_training = row["best_macrof1"]
        
    #     if macrof1_training == 0:
    #         continue

    #     config = row["config"]

    #     run_dir = os.path.join(root_dir, config)

    #     # find all files ending with .pt in the run_dir
    #     files = sorted([f for f in os.listdir(run_dir) if f.endswith(".pt")])
    #     best_model = files[-1]

    #     model = os.path.join(run_dir, best_model)

    #     model_names.append([f"{model}", index, beat_aligned, report_only, db_name, batch_size])


    p.map(run_validate_script, model_names, chunksize=1)


def merge_reports(root_dir, beat_aligned=True, db_name="ludb"):
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
                if beat_aligned and f"{db_name}/aligned/model_info" not in file_path:
                    continue
                elif not beat_aligned and f"{db_name}/fixed/model_info" not in file_path:
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
    merged_csv_path = os.path.join(root_dir, f"{db_name}_merged_{alignment}.csv")
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


    
def validation_summary_merger(root_dir, db_names, aligned=True):
    summary_csv = os.path.join(root_dir, "summary.csv")
    df = pd.read_csv(summary_csv, delimiter=";")
    new_df = df[["config", "total_params", "stopped_due_to"]].copy()

    for db_name in db_names:
        if aligned:
            path = f"{db_name}_merged_aligned"
        else:
            path = f"{db_name}_merged_fixed"
        # load merged csv file
        merged_csv_path = os.path.join(root_dir, f"{path}.csv")
        if not os.path.exists(merged_csv_path):
            print(f"❌ {merged_csv_path} does not exist.")
            continue
        df2 = pd.read_csv(merged_csv_path, delimiter=";")
        new_df[f"{db_name}__evaluation_metrics__tolerance_aware_classification_report__macro avg__f1-score"] = df2["evaluation_metrics__tolerance_aware_classification_report__macro avg__f1-score"]

    new_df.to_csv(os.path.join(root_dir, f"summary_{"_".join(db_names)}.csv"), index=False, sep=";")

    



if __name__ == "__main__":

    beat_aligned = True
    pad = True
    report_only = False
    filter_data = True
    batch_size = 32
    pool_size = 3

    root_dir = "results/hpo_multi_qrs_aligned2/BiXLSTM2new"
    #root_dir = "results/hpo_multi_qrs_aligned2_no_pad/BiXLSTM2new"

    data_dir = "lvad_ecgs_new"
    db_name = "lvad_ecgs_new"

    #data_dir = "QTDataset"
    #db_name = "qtdb"

    

    #data_dir = "ludb"
    #db_name = "ludb_filtered"
    #db_name = "rabbit_clean"

    data_dirs = ["lvad_ecgs_new", "ludb"]
    db_names = ["lvad_ecgs_new", "ludb_filtered"]



    

    #data_dirs = ["ludb_qrs_aligned_seq", "ludb"]

    data_dirs = ["ludb_qrs_aligned_padded"]
    db_names = ["ludb_qrs_aligned_padded"]


    data_dirs = [f"ludb_lead{i}_qrs_aligned_padded" for i in range(8, 12)]
    db_names = [f"ludb_lead{i}_qrs_aligned_padded" for i in range(8, 12)]

    # data_dirs = [f"ludb_lead{i}_qrs_aligned_unpadded" for i in range(2)]
    # db_names = [f"ludb_lead{i}_qrs_aligned_unpadded" for i in range(2)]

    # data_dirs = ["lvad_ecgs_new_qrs_aligned_padded"]
    # db_names = ["lvad_ecgs_new_qrs_aligned_padded"]

    # #data_dirs = ["ludb_qrs_aligned_seq"]
    # #db_names = ["ludb_qrs_aligned_seq"]

    data_dirs = ["rabbit_qrs_aligned_padded"]
    db_names = ["rabbit_qrs_aligned_padded"]
    
    #TODO: rerun with "ludb" (new)

    for data_dir, db_name in zip(data_dirs, db_names):
        validate(root_dir, 
                beat_aligned=beat_aligned, 
                pad=pad,
                report_only=report_only, 
                data_dir=data_dir, 
                db_name=db_name, 
                pool_size=pool_size, 
                batch_size=batch_size, 
                filter_data=filter_data)
        merge_reports(root_dir, beat_aligned=beat_aligned, db_name=db_name)

    #validate(root_dir, beat_aligned=False, report_only=True)


    #merge_reports(root_dir, beat_aligned=False)
    #validation_summary_merger(root_dir, ["ludb_filtered", "lvad_ecgs_new"], aligned=True)

