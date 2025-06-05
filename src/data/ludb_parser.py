import wfdb
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import copy

class LUDBParser:
    LEADS = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    SRC_FS = 500
    LABEL_TO_CLASS = {
        "NW": 0, # No wave
        "p": 1, # P-wave
        "N": 2,  # QRS complex
        "t": 3,  # T-wave
    }

    def __init__(self, data_dir="/home/david/CARDISENSE/Datasets/lobachevsky-university-electrocardiography-database-1.0.1/data", fs=250):
        self.data_dir = data_dir
        self.fs = fs
        # self.data = None
        # self.labels = None


    def load_data(self):
        # Load the data from the LUDB dataset
        data = {}
        labels = {}
        
        for rec_id in range(200):
            rec = os.path.join(self.data_dir, str(rec_id+1))
            rec_signal = wfdb.rdrecord(rec).p_signal
            rec_data = {}
            rec_labels = {}
            for (i, lead) in enumerate(self.LEADS):
                lead_data = rec_signal[:, i]
                lead_label_samples = wfdb.rdann(rec, extension = lead).sample
                lead_label_symbols = wfdb.rdann(rec, extension = lead).symbol
                
                if self.has_invalid_waves(rec_id, lead, lead_data, lead_label_symbols, lead_label_samples):
                    continue
                
                    
                
                lead_labels = self.create_lead_label_vector(lead_data, lead_label_symbols, lead_label_samples)
                lead_data, lead_labels = self.cut_unlabeled_data(lead_data, lead_labels, lead_label_samples)
                #lead_data, lead_labels = self.cut_incomplete_beats(rec_id, lead, lead_data, lead_labels)

                #self.has_incomplete_beats(rec_id, lead, lead_data, lead_labels)

                lead_data, lead_labels = self.downsample(lead_data, lead_labels)

                rec_data[f"lead{i}"] = lead_data
                rec_labels[f"lead{i}"] = lead_labels

            data[f"rec{rec_id}"] = rec_data
            labels[f"rec{rec_id}"] = rec_labels


                #rec_labels.append(lead_labels)
                #print(f"Loaded annotations for {lead} from {rec} with shape {rec_labels[-1].shape}")
            
            #print(f"Loaded signal from {rec} with shape {rec_data.shape}")

            
            #self.data.append(np.concatenate(rec_data, axis=0))
            #self.labels.append(np.concatenate(rec_labels, axis=0))

        
        #self.data = np.concatenate(self.data, axis=0)
        #self.labels = np.concatenate(self.labels, axis=0)

        return data, labels

    def has_invalid_waves(self, rec_id, lead, lead_data, lead_label_symbols, lead_label_samples):
        for i in range(0, len(lead_label_symbols)-2, 3):
            b = lead_label_symbols[i]
            p = lead_label_symbols[i+1]
            e = lead_label_symbols[i+2]
            if not (b == "(" and p in LUDBParser.LABEL_TO_CLASS.keys() and e == ")"):
                print(f"Warning: Invalid wave format {lead_label_symbols[i:i+3]} for {lead} in {rec_id}")
                return True
        return False


        


    def create_lead_label_vector(self, lead_data, lead_label_symbols, lead_label_samples):
        # Create a label vector for the lead
        lead_labels = np.zeros(len(lead_data), dtype=np.long)
        for i in range(0, len(lead_label_symbols)-2, 3):
            start = lead_label_samples[i]
            label = LUDBParser.LABEL_TO_CLASS[lead_label_symbols[i+1]]
            end = lead_label_samples[i+2]
            lead_labels[start:end] = label
        return lead_labels
    
    def cut_unlabeled_data(self, lead_data, lead_labels, lead_label_samples):
        # Cut the unlabeled data
        start = lead_label_samples[0]#-10
        end = lead_label_samples[-1]#+10
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
        start_index = np.where(lead_labels == LUDBParser.LABEL_TO_CLASS["p"])[0][0]
        # find index of last occurence of t-wave
        end_index = np.where(lead_labels == LUDBParser.LABEL_TO_CLASS["t"])[0][-1]
        return lead_data[start_index:end_index], lead_labels[start_index:end_index]

    def downsample(self, lead_data, lead_labels):
        if self.fs != LUDBParser.SRC_FS:
            downsample_factor = int(LUDBParser.SRC_FS / self.fs)
            lead_data = lead_data[::downsample_factor]
            lead_labels = lead_labels[::downsample_factor]
        return lead_data, lead_labels
    
    def cut_in_sequences(self, data, labels, seq_length = 300, start_wave = None, stride = None):
        x_seq = []
        y_seq = []
        if start_wave is not None:
            # Cut the data into sequences of length seq_length beginning with the start_label
            start_wave_class = LUDBParser.LABEL_TO_CLASS[start_wave]
            for rec in range(len(data)):
                x_rec = []
                y_rec = []
                for lead in range(len(data[rec])):
                    lead_data = data[rec][lead]
                    lead_labels = labels[rec][lead]
                    start_indices = np.where(lead_labels == start_wave_class)[0]
                    start_indices = start_indices[np.where(lead_labels[start_indices-1] != start_wave_class)[0]]

                    # transitions = np.diff(np.concatenate([[start_wave_class-1], start_wave_class])) == 1
                    # start_indices = np.where(lead_labels[transitions])[0]

                    # start_indices = np.where(lead_labels[start_indices-1] != start_wave_class)[0]

                    for start_index in start_indices:
                        if start_index + seq_length < len(lead_data):
                            x_rec.append(lead_data[start_index:start_index+seq_length])
                            y_rec.append(lead_labels[start_index:start_index+seq_length])
                x_seq.append(x_rec)
                y_seq.append(y_rec)
        else:
            if stride is None:
                stride = seq_length
            # Cut the data into sequences of length seq_length
            for rec in range(len(data)):
                x_rec = []
                y_rec = []
                for lead in range(len(data[rec])):
                    lead_data = data[rec][lead]
                    lead_labels = labels[rec][lead]
                    for start_index in range(0, len(lead_data)-seq_length, stride):
                        x_rec.append(lead_data[start_index:start_index+seq_length])
                        y_rec.append(lead_labels[start_index:start_index+seq_length])
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
            start_wave_class = LUDBParser.LABEL_TO_CLASS[start_wave]
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

                            if rec == "rec56" and lead == "lead0":
                                print(f"Padding sequence for rec {rec}, lead {lead}, start index {start_index} with next start index {start_indices[si+1]}")
                        
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
                x_train.append(np.array(data[train_index][lead]))
                y_train.append(np.array(labels[train_index][lead]))
        for val_index in val_indices:
            for lead in range(len(data[val_index])):
                x_val.append(np.array(data[val_index][lead]))
                y_val.append(np.array(labels[val_index][lead]))
        
        return x_train, y_train, x_val, y_val
    
    def get_dataloaders(self, x_train, y_train, x_val=None, y_val=None, batch_size=256, num_workers=4, pin_memory=True):
        # Create dataloaders for the training and validation sets
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        train_loader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=num_workers, 
                                  pin_memory=pin_memory)
        if x_val is None or y_val is None:
            val_loader = None
        else:
            x_val = np.array(x_val)
            y_val = np.array(y_val)
            val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
            val_loader = DataLoader(val_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=num_workers, 
                                    pin_memory=pin_memory)

        return train_loader, val_loader



ludb = LUDBParser()
data, labels = ludb.load_data()

assert len(data) == len(labels)
for r in data.keys():
    assert len(data[r]) == len(labels[r])
    for l in data[r].keys():
        assert len(data[r][l]) == len(labels[r][l])

#x_seq, y_seq = ludb.cut_in_sequences(data, labels, seq_length = 300, start_wave = None, stride = 50)

# len(data), len(labels)
# len(data[1]), len(labels[1])
# len(data[1][2]), len(labels[1][2])

# merged_x = []
# merged_y = []
# for rec in range(len(data)):
#     for lead in range(len(data[rec])):
#         merged_x += data[rec][lead].tolist()
#         merged_y += labels[rec][lead].tolist()

# assert(len(merged_x) == len(merged_y))

# np.savez_compressed("../DATA/ludb/fs250_x.npz", merged_x)
# np.savez_compressed("../DATA/ludb/fs250_y.npz", merged_y)


# plt.plot(data["rec10"]["lead0"]); plt.plot(labels["rec10"]["lead0"]); plt.show();

seq_data, seq_labels = ludb.cut_leads_in_sequences(
    copy.deepcopy(data), 
    copy.deepcopy(labels), 
    seq_length = 300, 
    start_wave = "N", 
    pad=True,
    pad_label=-100,
    pad_value=0)

for i in range(55,57):
    print(f"rec{i}")
    plt.plot(seq_data[f"rec{i}"]["lead0"][0]); plt.plot(seq_labels[f"rec{i}"]["lead0"][0]); plt.show();

import pickle

# with open('../DATA/ludb/fs250_x_seqs_qrs_aligned_padded.pickle', 'wb') as handle:
#     pickle.dump(seq_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('../DATA/ludb/fs250_y_seqs_qrs_aligned_padded.pickle', 'wb') as handle:
#     pickle.dump(seq_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


x_seq = []
y_seq = []
for rec in seq_data.keys():
    for lead in seq_data[rec].keys():
        for i in range(len(seq_data[rec][lead])):
            x_seq.append(seq_data[rec][lead][i])
            y_seq.append(seq_labels[rec][lead][i])

out_dir = os.path.join("..", "DATA", f"ludb_qrs_aligned_padded")
os.makedirs(out_dir, exist_ok=True)
with open(f'{out_dir}/fs250_x.pickle', 'wb') as handle:
    pickle.dump(x_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{out_dir}/fs250_y.pickle', 'wb') as handle:
    pickle.dump(y_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)

for i in range(20,30):
    plt.plot(x_seq[i]); plt.show();



seq_data, seq_labels = ludb.cut_leads_in_sequences(
    copy.deepcopy(data), 
    copy.deepcopy(labels), 
    seq_length = 300, 
    start_wave = "N", 
    pad=False)


for lead in range(12):
    x_seq_lead = []
    y_seq_lead = []
    for rec in seq_data.keys():
        if f"lead{lead}" not in seq_data[rec]:
            continue
        for i in range(len(seq_data[rec][f"lead{lead}"])):
            x_seq_lead.append(seq_data[rec][f"lead{lead}"][i])
            y_seq_lead.append(seq_labels[rec][f"lead{lead}"][i])
    out_dir = os.path.join("..", "DATA", f"ludb_lead{lead}_qrs_aligned_unpadded")
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/fs250_x.pickle', 'wb') as handle:
        pickle.dump(x_seq_lead, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{out_dir}/fs250_y.pickle', 'wb') as handle:
        pickle.dump(y_seq_lead, handle, protocol=pickle.HIGHEST_PROTOCOL)


#seq_data = {}
#seq_labels = {}
# for rec in range(len(_seq_data)):
#     seq_data[f"rec{rec}"] = {}
#     seq_labels[f"rec{rec}"] = {}
#     for lead in range(len(_seq_data[rec])):
#         seq_data[f"rec{rec}"][f"lead{lead}"] = []
#         seq_labels[f"rec{rec}"][f"lead{lead}"] = []
#         for s in range(len(_seq_data[rec][lead])):
#             seq_data[f"rec{rec}"][f"lead{lead}"].append(_seq_data[rec][lead][s])
#             seq_labels[f"rec{rec}"][f"lead{lead}"].append(_seq_labels[rec][lead][s])


plt.plot(seq_data["rec199"]["lead1"][0]); plt.plot(seq_labels["rec199"]["lead1"][0]); plt.show();
plt.plot(data[10][0]); plt.plot(labels[10][0]); plt.show();

import pickle
with open('../DATA/ludb/fs250_x_qrs_aligned_seq.pickle', 'wb') as handle:
    pickle.dump(seq_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../DATA/ludb/fs250_y_qrs_aligned_seq.pickle', 'wb') as handle:
    pickle.dump(seq_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


# with open('../DATA/ludb/fs250_x_seq.pickle', 'rb') as handle:
#     b = pickle.load(handle)

# lead_x = [[] for _ in range(12)]
# lead_y = [[] for _ in range(12)]
# for lead in range(12):
#     for rec in range(len(data)):
#         if lead >= len(data[rec]):
#             continue
#         lead_x[lead] += data[rec][lead].tolist()
#         lead_y[lead] += labels[rec][lead].tolist()

# assert(len(lead_x) == len(lead_y))
# for i in range(len(lead_x)):
#     assert(len(lead_x[i]) == len(lead_y[i]))

# for lead in range(len(lead_x)):
#     out_dir = os.path.join("..", "DATA", f"ludb_lead{lead+1}")
#     os.makedirs(out_dir, exist_ok=True)
#     np.savez_compressed(os.path.join(out_dir, "fs250_x.npz"), *lead_x[lead])
#     np.savez_compressed(os.path.join(out_dir, "fs250_y.npz"), *lead_y[lead])

        

import matplotlib.pyplot as plt
# # plt.plot(data[177][1])
# # plt.plot(labels[177][1])
# # plt.show()


# data, labels = ludb.cut_in_sequences(data, labels, seq_length = 300, start_wave = "N")

# # len(data), len(labels)
# # len(data[1]), len(labels[1])
# # len(data[1][1]), len(labels[1][1])

# # plt.plot(data[17][2])


# # plt.plot(labels[17][2])
# # plt.show()

# # x_train, y_train, x_val, y_val = ludb.split_sequences(data, labels, split_at=0.8, seed=42)
# # dataloader_train, dataloader_val = ludb.get_dataloaders(x_train, y_train, x_val, y_val, batch_size=256)
# # len(dataloader_train), len(dataloader_val)
# # xb, yb = next(iter(dataloader_train))
# # xb.shape, yb.shape