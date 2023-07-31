import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import neural_networks
import data_manager

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from torchmetrics.regression import MeanSquaredLogError, SpearmanCorrCoef

from colorama import Fore, Back, Style


data = pd.read_excel("/Users/baravidov/Desktop/Master/Project/Neural-Network/data_from_Ens_Grad.xlsx")
data = data.dropna().reset_index(drop = True)
data = data_manager.remove_missing_values(data)

#data = data.iloc[:400,:]


#spliting data to train and test and one hot encoding sequences
train_x, train_y, test_x, test_y, enrichment_factor = data_manager.data_manager_for_LSTM(data)

#creating batches for train and test
dataset_train = data_manager.Build_Data(train_x, train_y)
train_loader = DataLoader(dataset = dataset_train, batch_size = 64, drop_last = True)
dataset_test = data_manager.Build_Data(test_x, test_y)
test_loader = DataLoader(dataset = dataset_test, batch_size = 64, drop_last = True)

#model = neural_networks.CNN()
#model = neural_networks.CNNLSTM() #CNN and LSTM are combined into a feedforward net
model = neural_networks.LSTM() #CNN is feed into an LSTM and then a feedforward net
#model = neural_networks.LSTM(num_layers = 1)
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
num_epochs = 201

Fore.RED
for epoch in range(num_epochs):

    predictions = []
    targets = []

    total_loss = 0
    for x, y in train_loader:
        y_pred = model(x)
        loss = loss_fc(y_pred, y)
        total_loss += loss.item()
        predictions.extend(y_pred.tolist())
        targets.extend(y.tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(loss.item())

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
            y_pred = model(x)
            test_predictions.extend(y_pred.detach().numpy())
            test_labels.extend(y.detach().numpy())

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
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

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


torch.save(model.state_dict(), "~/model")
