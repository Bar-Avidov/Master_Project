import numpy as np
import pandas as pd
import torch
import statistics
import random


amino_acids = "ACDEFGHIKLMNPQRSTVWY"
amino_dict = {aa: i + 1 for i, aa in enumerate(amino_acids)}


#input: amino acid sequence
#output: tensor of indexes
def tokenize(sequence: str) -> list:
    indexes = []
    for aa in sequence:
        indexes.append(amino_dict[aa])

    return torch.tensor(indexes)


