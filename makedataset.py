import os
import gc

import json
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings(action='ignore')

LANDMARK_FILES_DIR = "./train_landmark_files"
TRAIN_FILE = "./train.csv"
label_map = json.load(open("./sign_to_prediction_index_map.json", "r"))

class FeatureGen(nn.Module):
    def __init__(self):
        super(FeatureGen, self).__init__()
        pass
    
    def forward(self, x):
        LEN = x.shape[0]

        return x.reshape(LEN, -1)
    
feature_converter = FeatureGen()

ROWS_PER_FRAME = 543
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

a = feature_converter(torch.tensor(load_relevant_data_subset('./train_landmark_files/26734/1000035562.parquet')))
DIM = a.shape[1]
DIM

# import multiprocessing as mp
from joblib import Parallel, delayed


def convert_row(path, label, participant_id):
    x = load_relevant_data_subset(os.path.join("./", path))
    return feature_converter(torch.tensor(x)).cpu().numpy(), label, participant_id

MAX_LEN = 256

def convert_and_save_data(val=False):
    df = pd.read_csv(TRAIN_FILE)
    df['label'] = df['sign'].map(label_map)

    if val:
        df=df[df.participant_id.isin([4718,55372,62590,37779,26734])]
    else:
        df=df[~df.participant_id.isin([4718,55372,62590,37779,26734])]

    data_list = []

    indexs = np.arange(df.shape[0],dtype=int)
    np.random.seed(1008600)
    np.random.shuffle(indexs)

    for i, (path, label, participant_id) in tqdm(enumerate(df[['path', 'label', 'participant_id']].values[indexs]), total=df.shape[0]):
        data, label, participant_id = convert_row(path, label, participant_id)
        if i==0:
            print(data.shape, label, data, participant_id)
        data_list.append({'data':data, 'label':label, 'participant_id': participant_id})

    if val:
        np.save(f"./val.npy", np.array(data_list))
    else:
        sublen = df.shape[0]//8+1
        for i in range(8):
            np.save(f"./train_{i}.npy", np.array(data_list[i*sublen:i*sublen+sublen]))

convert_and_save_data(True)
convert_and_save_data(False)

def convert_and_save_data2():
    df = pd.read_csv(TRAIN_FILE)
    df['label'] = df['sign'].map(label_map)

    data_list = []

    indexs = np.arange(df.shape[0],dtype=int)
    np.random.seed(68001)
    np.random.shuffle(indexs)

    for i, (path, label, participant_id) in tqdm(enumerate(df[['path', 'label', 'participant_id']].values[indexs]), total=df.shape[0]):
        data, label, participant_id = convert_row(path, label, participant_id)
        if i==0:
            print(data.shape, label, data, participant_id)
        data_list.append({'data':data, 'label':label, 'participant_id': participant_id})

    sublen = df.shape[0]//8+1
    for i in range(8):
        np.save(f"./atrain_{i}.npy", np.array(data_list[i*sublen:i*sublen+sublen]))

convert_and_save_data2()
