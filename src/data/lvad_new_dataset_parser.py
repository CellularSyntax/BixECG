import numpy as np
import os
import pickle
import copy
from validate_model import apply_bandpass_filter, z_normalize

def cut_leads_in_sequences(_data, _labels, seq_length = 300, start_wave = None, stride = None, pad=False, pad_label=-100, pad_value=0):
    #data = copy.deepcopy(_data)
    #labels = copy.deepcopy(_labels)
    data = {}
    labels = {}
    if start_wave is not None:
        # Cut the data into sequences of length seq_length beginning with the start_label
        start_wave_class = 2
        for rec in _data.keys():
            data[rec] = {}
            labels[rec] = {}
            for lead in _data[rec].keys():
                data[rec][lead] = []
                labels[rec][lead] = []

                lead_data = copy.deepcopy(_data[rec][lead])
                #lead_data = apply_bandpass_filter(lead_data, lowcut=0.5, highcut=50, fs=250, order=3)
                #lead_data = z_normalize(lead_data)
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


data_dir = "/home/david/Projects/DATA/lvad_ecgs_new"

# file0 = "HM3146_rawECG_20210705_fs250_combined.pkl"
# file0 = "HM3137_rawECG_20210318_fs250_combined.pkl"
# file0 = "HM3194_rawECG_20221212_fs250_combined.pkl"

# data = np.load(os.path.join(data_dir, file0), allow_pickle=True)

# print(data.keys())

data = {}
labels = {}

for (r, file) in enumerate(sorted(os.listdir(data_dir))):
    if not file.endswith(".pkl"):
        continue

    #rec_data = np.load(os.path.join(data_dir, file), allow_pickle=True)

    with open(os.path.join(data_dir, file), 'rb') as f:
        file_data = pickle.load(f)

    if len(file_data["labels"]) != len(file_data["ecg_c1"]):
        print(f"skipping {file}")
        continue
    
    if file_data["fs"] != 250:
        print(f"{file} fs: {file_data["fs"]}")

    rec_data = {}
    rec_labels = {}
    for (l, k) in enumerate(["ecg_c1", "ecg_c2", "ecg_c3"]):
        if k not in file_data:
            continue
        print(f"file {file}: len {k}: {len(file_data[k])} labels: {len(file_data['labels'])}")

        x = file_data[k].tolist()
        y = file_data["labels"].tolist()
        #x = apply_bandpass_filter(x, lowcut=0.5, highcut=40, fs=250, order=3).tolist()

        rec_data[f"lead{l}"] = x
        rec_labels[f"lead{l}"] = y

    data[f"rec{r}"] = rec_data
    labels[f"rec{r}"] = rec_labels



assert len(data) == len(labels)
for r in data.keys():
    assert len(data[r]) == len(labels[r])
    for l in data[r].keys():
        assert len(data[r][l]) == len(labels[r][l])


# merged_data = []
# merged_labels = []
# for rec in range(len(data)):
#     for lead in range(len(data[rec])):
#         merged_data += data[rec][lead]
#         merged_labels += labels[rec][lead]
        

# assert(len(merged_data) == len(merged_labels))

# np.savez_compressed(os.path.join(data_dir, "fs250_x.npz"), merged_data)
# np.savez_compressed(os.path.join(data_dir, "fs250_y.npz"), merged_labels)


import matplotlib.pyplot as plt
seq_data, seq_labels = cut_leads_in_sequences(
    copy.deepcopy(data), 
    copy.deepcopy(labels), 
    seq_length = 300, 
    start_wave = "N", 
    pad=True,
    pad_label=-100,
    pad_value=0)

plt.plot(np.array(data["rec15"]["lead0"][::1][0:300])); plt.plot(labels["rec15"]["lead0"][::1][0:300]); plt.show();

plt.plot(seq_data[f"rec0"]["lead0"][10]);  plt.show();

import pickle

x_seq = []
y_seq = []
for rec in seq_data.keys():
    for lead in seq_data[rec].keys():
        for i in range(len(seq_data[rec][lead])):
            x_seq.append(seq_data[rec][lead][i])
            y_seq.append(seq_labels[rec][lead][i])

print(f"len x_seq: {len(x_seq)}, len y_seq: {len(y_seq)}")
with open('../DATA/lvad_ecgs_new_qrs_aligned_padded/fs250_x.pickle', 'wb') as handle:
    pickle.dump(x_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../DATA/lvad_ecgs_new_qrs_aligned_padded/fs250_y.pickle', 'wb') as handle:
    pickle.dump(y_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('../DATA/lvad_ecgs_new_qrs_aligned_padded/fs250_x.pickle', 'rb') as handle:
    x_data = pickle.load(handle)
with open('../DATA/lvad_ecgs_new_qrs_aligned_padded/fs250_y.pickle', 'rb') as handle:
    y_data = pickle.load(handle)
x_seq = x_data
y_seq = y_data
x_seq = np.array([apply_bandpass_filter(seq, fs=250) for seq in x_seq])

plt.plot(z_normalize(apply_bandpass_filter(x_seq[1000][:300]))); plt.show();
plt.plot(x_seq[1000]); plt.show();


lead_data = [[] for _ in range(3)]
lead_labels = [[] for _ in range(3)]
for lead in range(3):
    for rec in range(len(data)):
        if lead > len(data[rec]) - 1:
            continue
        lead_data[lead] += data[rec][lead]
        lead_labels[lead] += labels[rec][lead]

assert(len(lead_data) == len(lead_labels))
for i in range(len(lead_data)):
    assert(len(lead_data[i]) == len(lead_labels[i]))

for lead in range(len(lead_data)):
    out_dir = os.path.join("..", "DATA", f"lvad_ecgs_new_lead{lead+1}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "fs250_x.npz"), lead_data[lead])
    np.savez_compressed(os.path.join(out_dir, "fs250_y.npz"), lead_labels[lead])




# import matplotlib.pyplot as plt
# plt.plot([s/1000 for s in data[10000:10300]])
# plt.plot(labels[10000:10300])
# plt.show()


# data_old = np.load("/home/david/Projects/DATA/lvad/fs250_x.npz")["arr_0"]
# plt.plot([s/1000 for s in data["ecg_c1"][0:600]])
# plt.plot([s/100 for s in apply_bandpass_filter(data["ecg_c1"][0:600], lowcut=0.5, highcut=40, fs=250, order=5)])
# plt.plot([s for s in z_normalize(apply_bandpass_filter(data["ecg_c1"][3000:3600], lowcut=0.5, highcut=40, fs=250, order=5))])
# plt.plot(data["labels"][3000:3600])
# plt.plot([s for s in data_old[0:600]])
# #plt.plot(labels[10000:10300])
# plt.show()
