import pandas as pd
import neural_networks
import torch

model = neural_networks.LSTM()

model.load_state_dict(torch.load("path/to/weights/file"))
amino_acids = ["A", "C", "D", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

#input: a data frame of sequences and corresponding enrichment scores
#output: tuple of two lists consisting 10 best sequences and corresponding enrichment scores
def load_best_sequences(data : pd.DataFrame):
    sorted_data = data.sort_values(by = 0)
    best_sequences = sorted_data.iloc[:10, :]
    return best_sequences

def affinity_maturation(sequence):
    len = len(sequence)
    sequences = []
    for i in range(len):
        for j in range(i+1, len):
            for k in range(20):
                new_sequence = sequence[:i-1] + amino_acids[k] + sequence[i+1:j-1] + amino_acids[k] + sequence[j+1:] 
                sequences.append(new_sequence)
    return sequences
