import pickle
import gzip
import torch
        
import os
from tqdm import tqdm
import numpy as np
from opencc import OpenCC

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object
    
def select_longest_strings(input_list, n=5):
    sorted_list = sorted(input_list, key=len, reverse=True)
    return sorted_list[:n]
      

data = pickle.load(open("./phoenix_iso.train", "rb"))
new_data = {}

for d in data:
    d['video_file'] = d['video_file'].replace("train/","")
    if d['video_file'] in new_data.keys():
        pass
    else:
        new_data[d['video_file']] = []
        
    new_data[d['video_file']].append(d)

data_pose = torch.load("./train_clean.pt")
target_pkl = './phoenix_gloss2pose_results.pkl'

target = {}

cc = OpenCC('s2t')


for key in tqdm(data_pose.keys()): 
    for gloss in new_data[key]:
        print(data_pose[key]['poses_3d'].shape)
        
        pose = np.array(data_pose[key]['poses_3d'][gloss['start']:gloss['end']])
        print(pose.shape)
        gloss['label'] = cc.convert(gloss['label']).upper()
        
        if gloss['label'] in target.keys():
            pass
        else:
            target[gloss['label']] = []
        
        if gloss['end'] - gloss['start'] <= 25:
            target[gloss['label']].append(pose)


for key in target.keys():
    target[key] = select_longest_strings(target[key])

print(len(target))

with open(target_pkl, 'wb') as f:
    pickle.dump(target, f)