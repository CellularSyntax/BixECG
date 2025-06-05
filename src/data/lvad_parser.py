import scipy
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

class LVADParser:

    SRC_FS = 250
    LABEL_TO_CLASS = {
        "NW": 0, # No wave
        "p": 1, # P-wave
        "N": 2,  # QRS complex
        "t": 3,  # T-wave
    }

    def __init__(self, data_dir="/home/david/CARDISENSE/Datasets/ECG_data/LVAD_ECGs/S01_labelled", fs=250):
        self.data_dir = data_dir
        self.fs = fs
        # self.data = None
        # self.labels = None

    def process_file(self, file):
        rec_data = []
        rec_labels = []

        print(file)
        df = pd.read_csv(file, sep=",")
        x = df["Signal"].tolist()
        y = df["Label"].tolist()
        
        for i in range(len(y)):
            if y[i] == 3:
                y[i] = 2
            elif y[i] == 2:
                y[i] = 3

        x = list(map(float, x)) 
        y = list(map(int, y)) 

        lead_data = x
        lead_labels = y

        #lead_data, lead_labels = self.downsample(lead_data, lead_labels)
        
        rec_data.append(lead_data)
        rec_labels.append(lead_labels)

        return rec_data, rec_labels
    

    def downsample(self, lead_data, lead_labels):
        if self.fs != LVADParser.SRC_FS:
            downsample_factor = int(LVADParser.SRC_FS / self.fs)
            lead_data = lead_data[::downsample_factor]
            lead_labels = lead_labels[::downsample_factor]
        return lead_data, lead_labels


    def load_data(self, proccess_pool_size=5):
        # Load the data from the LUDB dataset
        data = []
        labels = []

        mat_files = []
        
        for (i, file) in enumerate(sorted(os.listdir(self.data_dir))):
            if  file.endswith(".csv"):
                print(f"Loading {file}")
                mat_files.append(f"{os.path.join(self.data_dir, file)}")

        # call function in parallel using proces pool. save results ordered
        from multiprocessing import Pool
        with Pool(processes=proccess_pool_size) as pool:
            results = pool.map(self.process_file, mat_files, chunksize=1)
            for i in range(len(results)):
                rec_data, rec_labels = results[i]
                data.append(rec_data)
                labels.append(rec_labels)

        print("num records: ", len(data))
        for i in range(len(data)):
            print(f"record {i}: {len(data[i])} leads")
            for j in range(len(data[i])):
                print(f"lead {j}: {len(data[i][j])} samples, {len(labels[i][j])} labels")


        return data, labels


lvad = LVADParser()
data, labels = lvad.load_data()
data_x = []
data_y = []
for rec in range(len(data)):
    for lead in range(len(data[rec])):
        data_x += data[rec][lead]
        data_y += labels[rec][lead]



np.savez_compressed("/home/david/Projects/DATA/lvad/fs250_x.npz", data_x)
np.savez_compressed("/home/david/Projects/DATA/lvad/fs250_y.npz", data_y)