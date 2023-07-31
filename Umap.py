import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import umap.umap_ as umap
import umap.plot as uplot
import neural_networks
import data_manager
from torch.nn.utils.rnn import pad_sequence
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio import Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from torchmetrics.regression import MeanSquaredLogError, SpearmanCorrCoef
import pandas as pd

from transformers import BertForMaskedLM, BertTokenizer, pipeline

#load data from panning

data_heavy = pd.read_excel("/Users/baravidov/Desktop/Master/Master/Data/heavy_4.xlsx")
data_light = pd.read_excel("/Users/baravidov/Desktop/Master/Master/Data/light_4.xlsx")

data_heavy = data_heavy.iloc[:,1].unique()
data_light = data_light.iloc[:,1].unique()

#load itay's clones
data_clones = pd.read_excel("/Users/baravidov/Desktop/Master/Project/data/Itay CDR sequences against RBD.xlsx")
data_clones = list(data_clones.iloc[:,6])
"""new_sequence = data_clones[10]
new_sequence = new_sequence[:2] + "A" + new_sequence[4:]
data_clones[10] = new_sequence"""
sequences = []
for i, sequence in enumerate(data_clones):
    sequences.append(sequence)

sequences = sequences[:1000]

#check for CDR sequences found in both heavy and light chain
for i, sequence in enumerate(data_heavy):
    if sequence in data_light:
        sequences.append(sequence)

#initializing ProtBERT model and it's Tokenizer
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")

spaces_sequences = []
for i, sequence in enumerate(sequences):
    spaces_sequences.append(" ".join(sequence))


spaced_sequences = spaces_sequences[:900]

tokens = tokenizer(spaced_sequences, padding = True, return_tensors = 'pt')
print(tokens.data['input_ids'].shape)
output = model(**tokens)
print(output.logits.shape)

matrix = output.logits.flatten(1,2).detach().numpy()

reduced_embeddings = umap.UMAP(n_components=2, n_neighbors = 50).fit(matrix)
specific_indices = list(range(0, 11))
uplot.diagnostic(reduced_embeddings, point_size = 10)
annotations = data_clones

for i in range(len(specific_indices)):
    x = reduced_embeddings.embedding_[i][0]
    y = reduced_embeddings.embedding_[i][1]
    annotation = str(i) + " " + str(annotations[i])
    plt.scatter(x = x, y = y, s = 30, color = 'red', label = "Itay's clone")
    plt.annotate(str(i), xy = (x, y), fontsize = 8)

plt.show()