import torch
import os
import pickle
import gzip

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

data = load_dataset_file("phoenix-2014t.test")

idx = 0
# text data
with open('./secretTestSet500.txt', 'r', encoding='utf-8') as file:
    for line in file:
        data[idx]['text'] = line.strip()
        idx+=1
        

with gzip.open('phoenix-2014t.test', 'wb') as f:
    pickle.dump(data, f)