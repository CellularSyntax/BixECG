import scipy
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.helper_fns import apply_bandpass_filter, z_normalize

from sklearn.preprocessing import MinMaxScaler

import copy

class RabbitParser:
    SRC_FS = 1000
    CLASSES = ['NW', 'p', 'N', 't']
    LABEL_TO_CLASS = {
        "NW": 0, # No wave
        "p": 1, # P-wave
        "N": 2,  # QRS complex
        "t": 3,  # T-wave
    }

    def __init__(self, data_dir="/home/david/CARDISENSE/Datasets/ECG_data/Rabbit_ECGs", fs=250):
        self.data_dir = data_dir
        self.fs = fs
        # self.data = None
        # self.labels = None


    def process_recording(self, rec, clean=False):
        rec_data = {}
        rec_labels = {}

        ecg = scipy.io.loadmat(rec).get("ecg")
        

        x = ecg[0][0][1][:, 0].tolist()
        y = ecg[0][0][3][0, :].tolist()
        length = min(len(x), len(y))
        x = x[:length]
        y = y[:length]

        x = list(map(float, x)) 
        y = list(map(int, y)) 

        for i in range(len(y)):
            if y[i] == 1 or y[i] == 2:
                y[i] -= 1
            elif y[i] == 4:
                y[i] = 2

        x, y = self.downsample(x, y)

        if clean:
            x = apply_bandpass_filter(x, lowcut=0.5, highcut=50, fs=250, order=5).tolist()
            #lead_data = z_normalize(lead_data).tolist()
            scaler = MinMaxScaler(feature_range=(-1, 1))
            x = scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten().tolist()

        
        rec_data["lead0"] = x
        rec_labels["lead0"] = y

        return rec_data, rec_labels



    def load_data(self, pool_size=10):
        # Load the data from the LUDB dataset
        data = {}
        labels = {}

        recordings = []
        for (i, file) in enumerate(sorted(os.listdir(self.data_dir))):
            if file.endswith(".mat"):
                recordings.append(os.path.join(self.data_dir, file))

        # Use multiprocessing to load the recordings in parallel
        from multiprocessing import Pool
        with Pool(processes=pool_size) as pool:
            results = pool.map(self.process_recording, recordings)
        for i, (rec_data, rec_labels) in enumerate(results):
            data["rec" + str(i)] = rec_data
            labels["rec" + str(i)] = rec_labels

        return data, labels
    
    
    def cut_incomplete_waves(self, lead_label_symbols, lead_label_samples):
        i = 0
        while i < len(lead_label_symbols):
            if (lead_label_symbols[i] == "(" and \
                lead_label_symbols[i+1] in RabbitParser.CLASSES and \
                lead_label_symbols[i+2] == ")"):
                break
            i += 1
        lead_label_symbols = lead_label_symbols[i:]
        lead_label_samples = lead_label_samples[i:]

        i = len(lead_label_symbols) - 1
        while i > 0:
            if (lead_label_symbols[i] == ")" and \
                lead_label_symbols[i-1] in RabbitParser.CLASSES and \
                lead_label_symbols[i-2] == "("):
                break
            i -= 1
        lead_label_symbols = lead_label_symbols[i:]
        lead_label_samples = lead_label_samples[i:]

        return lead_label_symbols, lead_label_samples
    

    def has_invalid_waves(self, rec, lead, lead_data, lead_label_symbols, lead_label_samples):
        for i in range(0, len(lead_label_symbols)-2, 3):
            s = lead_label_symbols[i]
            p = lead_label_symbols[i+1]
            e = lead_label_symbols[i+2]
            if not (s == "(" and p in RabbitParser.CLASSES and e == ")"):
                print(f"Warning: Invalid wave format {lead_label_symbols[i:i+3]} for {lead} in {rec}")
                return True
        return False


        


    def create_lead_label_vector(self, lead_data, lead_label_symbols, lead_label_samples):
        # Create a label vector for the lead
        lead_labels = np.zeros(len(lead_data), dtype=np.long)
        for i in range(0, len(lead_label_symbols)-2, 3):
            start = lead_label_samples[i]
            label = RabbitParser.LABEL_TO_CLASS[lead_label_symbols[i+1]]
            end = lead_label_samples[i+2]
            lead_labels[start:end] = label
        return lead_labels
    
    def cut_unlabeled_data(self, lead_data, lead_labels, lead_label_samples):
        # Cut the unlabeled data
        start = lead_label_samples[0]-10
        end = lead_label_samples[-1]+10
        return lead_data[start:end], lead_labels[start:end]
    
    def cut_incomplete_beats(self, rec_id, lead, lead_data, lead_labels):
        # Cut beginning until start of first p-wave and end until end of last t-wave
        # find index of first occurence of p-wave
        # print(f"Lead data length: {len(lead_data)}")
        # print(f"Lead labels length: {len(lead_labels)}")
        # print(f"Lead: {lead}")
        # print(f"Rec ID: {rec_id}")
        # print(0 in lead_labels)
        # print(1 in lead_labels)
        # print(2 in lead_labels)
        # print(3 in lead_labels)
        start_index = np.where(lead_labels == RabbitParser.LABEL_TO_CLASS["p"])[0][0]
        # find index of last occurence of t-wave
        end_index = np.where(lead_labels == RabbitParser.LABEL_TO_CLASS["t"])[0][-1]
        return lead_data[start_index-5:end_index+5], lead_labels[start_index-5:end_index+5]

    def downsample(self, lead_data, lead_labels):
        if self.fs != RabbitParser.SRC_FS:
            downsample_factor = int(RabbitParser.SRC_FS / self.fs)
            lead_data = lead_data[::downsample_factor]
            lead_labels = lead_labels[::downsample_factor]
        return lead_data, lead_labels
    
    def cut_in_sequences(self, data, labels, seq_length = 300, start_wave = None):
        x_seq = []
        y_seq = []
        if start_wave is not None:
            # Cut the data into sequences of length seq_length beginning with the start_label
            start_wave_class = RabbitParser.LABEL_TO_CLASS[start_wave]
            for rec in range(len(data)):
                x_rec = []
                y_rec = []
                for lead in range(len(data[rec])):
                    lead_data = data[rec][lead]
                    lead_labels = labels[rec][lead]
                    start_indices = np.where(lead_labels == start_wave_class)[0]
                    for start_index in start_indices:
                        if start_index + seq_length < len(lead_data):
                            x_rec.append(lead_data[start_index-5:start_index+seq_length-5])
                            y_rec.append(lead_labels[start_index-5:start_index+seq_length-5])
                x_seq.append(x_rec)
                y_seq.append(y_rec)

        data = x_seq
        labels = y_seq

        return data, labels
    

    def cut_leads_in_sequences(self, _data, _labels, seq_length = 300, start_wave = None, stride = None, pad=False, pad_label=-100, pad_value=0):
        #data = copy.deepcopy(_data)
        #labels = copy.deepcopy(_labels)
        data = {}
        labels = {}
        if start_wave is not None:
            # Cut the data into sequences of length seq_length beginning with the start_label
            start_wave_class = RabbitParser.LABEL_TO_CLASS[start_wave]
            for rec in _data.keys():
                data[rec] = {}
                labels[rec] = {}
                for lead in _data[rec].keys():
                    data[rec][lead] = []
                    labels[rec][lead] = []

                    lead_data = copy.deepcopy(_data[rec][lead])
                    lead_labels = np.concatenate((np.array([-1]), copy.deepcopy(_labels[rec][lead])))
                
                    start_indices = np.where(lead_labels == start_wave_class)[0]
                    start_indices = start_indices[np.where( (lead_labels)[start_indices-1] != start_wave_class)[0]] - 1
                    lead_labels = lead_labels[1:]

                   
                    #transitions = np.diff(np.concatenate([[start_wave_class-1], start_wave_class])) == 1
                    #start_indices = np.where(lead_labels[transitions])[0]

                    # start_indices = np.where(lead_labels[start_indices-1] != start_wave_class)[0]
                    #print(start_indices)
                    for (si, start_index) in enumerate(start_indices):
                        if start_index + seq_length >= len(lead_data):
                            break

                        x = np.array(copy.deepcopy(lead_data[start_index:start_index+seq_length]))
                        y = np.array(copy.deepcopy(lead_labels[start_index:start_index+seq_length]))

                        if pad and (si < len(start_indices)-1) and (start_indices[si+1] < start_index + seq_length):
                            # pad the sequence with zeros if the next start index is not far enough
                            #print(f"Padding sequence for rec {rec}, lead {lead}, start index {start_index} with next start index {start_indices[si+1]}")
                            x[start_indices[si+1]-start_indices[si]:] = pad_value
                            y[start_indices[si+1]-start_indices[si]:] = pad_label

                        data[rec][lead].append(x) 
                        labels[rec][lead].append(y)

        return data, labels


    def split_sequences(self, data, labels, split_at=0.8, seed=42):
        # Split the data into training and validation sets
        x_train = []
        y_train = []
        x_val = []
        y_val = []

        np.random.seed(seed)
        # shuffle indices
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        train_indices = indices[:int(len(data) * split_at)]
        val_indices = indices[int(len(data) * split_at):]

        # append all data to x_train and y_train using 
        for train_index in train_indices:
            for lead in range(len(data[train_index])):
                x_train.append(np.array(data[train_index][lead]).reshape(-1, 1))
                y_train.append(np.array(labels[train_index][lead]).reshape(-1, 1))
        for val_index in val_indices:
            for lead in range(len(data[val_index])):
                x_val.append(np.array(data[val_index][lead]).reshape(-1, 1))
                y_val.append(np.array(labels[val_index][lead]).reshape(-1, 1))
        
        return x_train, y_train, x_val, y_val
    
    def get_dataloaders(self, x_train, y_train, x_val=None, y_val=None, batch_size=256, num_workers=4, pin_memory=True):
        # Create dataloaders for the training and validation sets
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)

        train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))

        train_loader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=num_workers, 
                                  pin_memory=pin_memory)
        if x_val is None or y_val is None:
            val_loader = None
        else:
            val_loader = DataLoader(val_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=num_workers, 
                                    pin_memory=pin_memory)

        return train_loader, val_loader



rabbit = RabbitParser()
data, labels = rabbit.load_data()

len(data), len(labels)
len(data[0]), len(labels[0])
len(data[0][0]), len(labels[0][0])

# data_x, data_y = [], []
# for i in range(len(data)):
#     for j in range(len(data[i])):
#         data_x += (data[i][j])
#         data_y += (labels[i][j])

# len(data_x), len(data_y)
# np.savez_compressed("../DATA/rabbit/fs250_x.npz", data_x)
# np.savez_compressed("../DATA/rabbit/fs250_y.npz", data_y)

import matplotlib.pyplot as plt
plt.plot(data[0][0][0:300])
# plt.plot(data_y[0:300])
plt.show()

import pickle


seq_data, seq_labels = rabbit.cut_leads_in_sequences(
    copy.deepcopy(data), 
    copy.deepcopy(labels), 
    seq_length = 300, 
    start_wave = "N", 
    pad=True,
    pad_label=-100,
    pad_value=0)

x_seq = []
y_seq = []
for rec in seq_data.keys():
    for lead in seq_data[rec].keys():
        for i in range(len(seq_data[rec][lead])):
            x_seq.append(seq_data[rec][lead][i])
            y_seq.append(seq_labels[rec][lead][i])

out_dir = os.path.join("..", "DATA", f"rabbit_qrs_aligned_padded")
os.makedirs(out_dir, exist_ok=True)
with open(f'{out_dir}/fs250_x.pickle', 'wb') as handle:
    pickle.dump(x_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{out_dir}/fs250_y.pickle', 'wb') as handle:
    pickle.dump(y_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)



# plt.plot(data[177][1])
# plt.plot(labels[177][1])
# plt.show()

# data, labels = ludb.cut_in_sequences(data, labels, seq_length = 300, start_wave = "N")

# len(data), len(labels)
# len(data[1]), len(labels[1])
# len(data[1][1]), len(labels[1][1])

# plt.plot(data[17][2])


# plt.plot(labels[17][2])
# plt.show()

# x_train, y_train, x_val, y_val = ludb.split_sequences(data, labels, split_at=0.8, seed=42)
# dataloader_train, dataloader_val = ludb.get_dataloaders(x_train, y_train, x_val, y_val, batch_size=256)
# len(dataloader_train), len(dataloader_val)
# xb, yb = next(iter(dataloader_train))
# xb.shape, yb.shape