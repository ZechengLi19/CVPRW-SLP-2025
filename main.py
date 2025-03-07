import csv
import torch
import os
import torch
import zipfile
import os
import pickle
import gzip


def extract_second_column(file_path):
    second_column = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            try:
                second_column.append(float(row[1]))
            except ValueError:
                continue
    
    return second_column

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

file_path = './frame_lengths.csv' 
text_len_list = extract_second_column(file_path)

# Text to gloss data
gloss_data = pickle.load(open("./phoenix_text2gloss_results.pkl", "rb"))

# gloss pose data
gloss_support = pickle.load(open("./phoenix_gloss2pose_results.pkl", "rb"))

sentences = []
val_flag = torch.zeros(500)
pose_3d = []

for i in range(500):
    pose_3d.append(None)

# text data
with open('./secretTestSet500.txt', 'r', encoding='utf-8') as file:
    for line in file:
        sentences.append(line.strip())

# search extist gt pose in dataset
gt_label = load_dataset_file(f"./phoenix-2014t.train")
pose = torch.load("./train.pt")

for item in gt_label:
    if item['text'][:-2] in sentences:
        for idx, sen in enumerate(sentences):
            # find same sentence
            if sen == item['text'][:-2]:
                if val_flag[idx] == 1:
                    continue
                else:
                    val_flag[idx] = 1
                    
                    try:
                        pose_3d[idx] = pose[item['name'].split("/")[-1]]['poses_3d']
                    except:
                        val_flag[idx] = 0
                        print("fail to set 1")
                        
print(val_flag.sum())

# data for eval 
new_data = {}
gloss_data_list = list(gloss_data.keys())

for idx, key in enumerate(range(500)):
    # initial
    new_data[f"sentence_{idx}"] = torch.zeros((int(text_len_list[idx]), 178, 3))
    # replace by dataset sample
    if val_flag[idx] == 1:
        # match target len
        if pose_3d[idx].shape[0] > int(text_len_list[idx]):
            new_data[f"sentence_{idx}"] = pose_3d[idx][:int(text_len_list[idx])]
        
        else:
            new_data[f"sentence_{idx}"][:pose_3d[idx].shape[0]] = pose_3d[idx]
        
    else:
        # retrieval
        gloss_support_list = []
        for gls in gloss_data[gloss_data_list[idx]]['gls_hyp'].split(" "):
            if gls in gloss_support.keys():
                gloss_support_list.append(torch.tensor(gloss_support[gls][0]))
        
        if len(gloss_support_list) == 0:
            print("empty")
            continue
        
        # concat all gloss pose
        gloss_support_list = torch.concat(gloss_support_list, dim=0)
        
        # match target len
        if gloss_support_list.shape[0] > int(text_len_list[idx]):
            gloss_support_list = gloss_support_list[:int(text_len_list[idx])]
            new_data[f"sentence_{idx}"] = gloss_support_list
        
        else:
            new_data[f"sentence_{idx}"][:gloss_support_list.shape[0]] = gloss_support_list
    
torch.save(new_data, "./SLP_test.pt")

with zipfile.ZipFile("./SLP_test.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("./SLP_test.pt", os.path.basename('SLP_test.pt'))
