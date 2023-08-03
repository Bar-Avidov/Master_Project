#Packages for deep learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import BertTokenizer, pipeline
from torch.utils.data import DataLoader

#Personal modules
import neural_networks
import data_manager

#Packages for evaluation of results
from sklearn.metrics import r2_score
from torchmetrics.regression import SpearmanCorrCoef

#Additional useful packages
import time
from tqdm import tqdm



print("starting program!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} for LSTM model, bert model is computed on CPU", device.type)
#read excel file
data = pd.read_excel("~/Project/Neural-Network/data_from_Ens_Grad.xlsx")
data = data.dropna().reset_index(drop = True)
data = data_manager.remove_missing_values(data)

data = data.iloc[:1000,:]
print("number of sequences :", len(data))

#spliting data to train and test and one hot encoding sequences
train_x, test_x, train_y, test_y, enrichment_factor = data_manager.preprocessing_for_Bert(data)

print("data splitted into training and testing")

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
embeddings_pipeline = pipeline(task = "feature-extraction", model = "Rostlab/prot_bert", tokenizer = "Rostlab/prot_bert", framework = 'pt')

#embedding sequence using batches of size 64
embeddings_train_x = []
embeddings_test_x = []
dataset_sequences_train = data_manager.DataSetSequences(train_x)
train_sequences_loader = DataLoader(dataset = dataset_sequences_train, batch_size = 64, drop_last = True)
dataset_sequences_test = data_manager.DataSetSequences(test_x)
test_sequences_loader = DataLoader(dataset = dataset_sequences_test, batch_size = 64, drop_last = True)

start = time.time()
print("starting embedding")

for sequences in tqdm(train_sequences_loader):
    embeddings = embeddings_pipeline(sequences)
    embeddings_train_x.extend(embeddings)

end = time.time()

print("Finished embedding training set: ", end - start, "s")
start = time.time()

for sequences in tqdm(test_sequences_loader):
    embeddings = embeddings_pipeline(sequences)
    embeddings_test_x.extend(embeddings)

end = time.time()
print(f"Finished embedding test set: {end - start}s")

input_train_x = torch.tensor(embeddings_train_x).squeeze()
input_test_x = torch.tensor(embeddings_test_x).squeeze()
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)

max_length = input_train_x.shape[1]



#creating batches for train and test
dataset_train = data_manager.Build_Data(input_train_x, train_y)
train_loader = DataLoader(dataset = dataset_train, batch_size = 64, drop_last = True)
dataset_test = data_manager.Build_Data(input_test_x, test_y)
test_loader = DataLoader(dataset = dataset_test, batch_size = 64, drop_last = True)

model = neural_networks.Bert(max_length = max_length).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, weight_decay = 1e-5)
loss_fc = nn.MSELoss()
spearman = SpearmanCorrCoef()



#used to keep track of predictions during each epoch
results_df = pd.DataFrame()

torch.manual_seed(15)

r2_scores_train = []
r2_score_test = []
spearman_train_list = []
spearman_test_list = []
epoch_list = []
losses = []
num_epochs = 501
print("starting training loop")
for epoch in tqdm(range(num_epochs)):

    predictions = []
    targets = []

    total_loss = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model.forward(x)
        loss = loss_fc(y_pred, y)
        total_loss += loss.item()
        predictions.extend(y_pred.tolist())
        targets.extend(y.tolist())

        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()
    losses.append(total_loss)

    train_score = r2_score(targets, predictions)
    spearman_score = spearman(torch.tensor(predictions), torch.tensor(targets)).item()
    r2_scores_train.append(train_score)
    spearman_train_list.append(spearman_score)
    epoch_list.append(epoch)
    #pd.concat((results_df, pd.DataFrame(predictions)), axis = 1)
    #results_df['epoch' + str(epoch)] = predictions

    print("epochs number: ", epoch, "--------------------------------------")
    print("loss is: ", loss.item())
    print("training r2 score is: ", train_score)
    print("spearman correlation: ", spearman_score)
    
    if epoch % 10 == 0:

        test_predictions = []
        test_labels = []

        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model.forward(x)
            test_predictions.extend(y_pred.tolist())
            test_labels.extend(y.tolist())

        test_score = r2_score(test_labels, test_predictions)
        r2_score_test.append(test_score)
        spearman_test_score = spearman(torch.tensor(test_predictions), torch.tensor(test_labels)).item()
        spearman_test_list.append(spearman_test_score)

        print("testing r2 score: ", test_score)
        print("testing spearman correlation: ", spearman_test_score)

        if epoch % 50 == 0:
            plt.plot(r2_scores_train)
            plt.title("r2 score of training set")
            plt.xlabel("epochs")
            plt.ylabel("r2 scores")
            plt.show()

            plt.plot(r2_score_test)
            plt.title("r2 score of testing set")
            plt.xlabel("epochs")
            plt.ylabel("r2 scores")
            plt.show()
    
#df = pd.DataFrame(test_predictions, test_labels)
#df.to_csv("test results.csv")

test_predictions = []
test_labels = []
    

#results_df['labels'] = targets

plt.plot(losses)
plt.title("Loss function over Epochs")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig("/Home/Figures/losses.png")

plt.plot(r2_scores_train)
plt.title("r2 score of training set")
plt.xlabel("epochs")
plt.ylabel("r2 scores")
plt.savefig("/Home/Figures/training_r2_score.png")

plt.plot(r2_score_test)
plt.title("r2 score of testing set")
plt.xlabel("epochs")
plt.ylabel("r2 scores")
plt.savefig("/Home/Figures/testing_r2_score.png")

#plotting ground trutch enrichment (x) and predicted enrichment (y)
#can add for training and for testing
plt.plot(targets, predictions)
plt.title("Ground truth enrichment score and Predicted enrichment score")
plt.xlabel("Ground truth enrichment score")
plt.ylabel("Predictd enrichment score")
plt.savefig("/Home/Figures/train_Truth_vs_Predicted.png")