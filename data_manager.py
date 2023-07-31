import torch
import utils
import numpy as np
from sklearn.model_selection import train_test_split
from torch. utils. data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from transformers import BertTokenizer
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer


###########################################################################################


class DataSetSequences(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __getitem__(self, index):
        return self.sequences[index]
    
    def __len__(self):
        return len(self.sequences)

class DataSetForBert(Dataset):
    def __init__(self, sequences, scores):

        self.sequences = sequences
        self.score = scores

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.len = len(self.sequences)

    def __getitem__(self, index):
        
        sequence = self.sequences[index]
        score = self.score[index]

        inputs = self.tokenizer.encode_plus(sequence, add_special_tokens=True, padding='max_length', truncation=True, max_length=128)

        input_ids = torch.tensor(inputs['input_ids'])
        attention_mask = torch.tensor(inputs['attention_mask'])
        token_type_ids = torch.tensor(inputs['token_type_ids'])
        score = torch.tensor(score, dtype = torch.float32)

        return input_ids, attention_mask, token_type_ids, score
    

    def __len__(self):
        return self.len

###########################################################################################
        
def preprocessing_for_Bert(data : pd.DataFrame):

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")

    enrichment_factor = data.iloc[:,1].abs().max()
    data.iloc[:,1] = data.iloc[:,1] / enrichment_factor
    features = data.iloc[:,0].tolist()
    targets = data.iloc[:,1].tolist()

    max_seq_length = max(len(feature) for feature in features)

    for i, sequence in enumerate(features):

        seq_length = len(sequence)
        padding_length = max_seq_length - seq_length
        features[i] = " ".join(sequence) + (" " + tokenizer.pad_token) * padding_length

    train_x, test_x, train_y, test_y = train_test_split(features, targets, train_size = 0.8)

    return train_x, test_x, train_y, test_y, enrichment_factor

def data_manager_for_LSTM(data : pd.DataFrame):

    enrichment_factor = data.iloc[:,1].abs().max()
    data.iloc[:,1] = data.iloc[:,1] / enrichment_factor

    features = data.iloc[:,0].tolist()
    targets = data.iloc[:,1].tolist()
    

    for i, sequence in enumerate(features):
        features[i] = " ".join(sequence)

    #split all data to training and testing
    train_x, test_x, train_y, test_y = train_test_split(features, targets, train_size = 0.8)
    return train_x, train_y, test_x, test_y, enrichment_factor

def remove_missing_values(data : pd.DataFrame):

    indexes = []
    for i in range(len(data)):
        if 'X' in data.iloc[i,0]:
            indexes.append(i)
    
    data = data.drop(index = indexes, axis = 0)
    return data

class Build_Data(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = torch.tensor(y)
        self.len = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
