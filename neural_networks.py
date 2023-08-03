#Packages for deep learning
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Bert(nn.Module):
    def __init__(self, max_length, hidden_size = 1024, num_layers = 3):
        super (Bert, self).__init__()

        """
        #linear version
        self.linear1 = nn.Linear(1024, 1)
        self.linear2 = nn.Linear(max_length,1)
        """

        #LSTM version
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(1024, self.hidden_size, self.num_layers, batch_first = True, dropout = 0.2, bidirectional = True)
        self.linear1 = nn.Linear(int(self.hidden_size * 2), 1)
        self.linear2 = nn.Linear(max_length, 128)
        self.linear3 = nn.Linear(128, 1)

    #the model's input is: a str repressenting the amino acid sequence
    #since tokenizing and embedding is happening within the model
    def forward(self, x : torch.tensor):
        """
        #linear version
        output = torch.tanh(self.linear1(x))
        output = output.squeeze()
        output = torch.tanh(self.linear2(output))
        output = output.squeeze()

        return output
        """
        h0 = torch.zeros(2 * self.num_layers, x.shape[0], self.hidden_size).to(device)
        c0 = torch.zeros(2 * self.num_layers, x.shape[0], self.hidden_size).to(device)

        output, _ = self.lstm(x, (h0, c0))
        output = torch.tanh(self.linear1(output))
        output = output.squeeze(2)
        output = torch.tanh(self.linear2(output))
        output = torch.tanh(self.linear3(output))

        return output.squeeze(1)



class LSTM(nn.Module):
    def __init__(self, vocab_size = 21, embedding_size = 24, hidden_size = 64, num_layers = 1):
        super (LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx = 0)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first = True, dropout = 0.2, bidirectional = True)
        self.linear1 = nn.Linear(int(self.hidden_size * 2), 1)
        self.linear2 = nn.Linear(18, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, x):
        

        embedded = self.embedding(x)

        h0 = torch.zeros(2 * self.num_layers, embedded.shape[0], self.hidden_size)
        c0 = torch.zeros(2 * self.num_layers, embedded.shape[0], self.hidden_size)

        output, _ = self.lstm(embedded, (h0, c0))
        output = torch.tanh(self.linear1(output))
        output = output.squeeze(2)
        output = torch.tanh(self.linear2(output))
        output = torch.tanh(self.linear3(output))

        return output.squeeze(1)
    
class CNNLSTM(nn.Module):
    def __init__(self, vocab_size = 21, input_size = 24, embedding_size = 24, hidden_size = 64, num_layers = 1):
        super(CNNLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        #Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx = 0)
        
        #CNN
        self.Conv1 = nn.Conv1d(in_channels = self.embedding_size, out_channels = 64,kernel_size = 3, stride = 1)
        self.dropout1 = nn.Dropout(0.5)
        self.maxpool1 = nn.MaxPool1d(kernel_size = 3, stride = 1)
        self.Conv2 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 1)
        self.dropout2 = nn.Dropout(0.5)
        self.maxpool2 = nn.MaxPool1d(kernel_size = 3, stride = 1)
        self.Conv3 = nn.Conv1d(in_channels = 128, out_channels = 1, kernel_size = 3, stride = 1)
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6, 1)

        #LSTM
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True, dropout = 0.2, bidirectional = True)
        self.linear1 = nn.Linear(int(self.hidden_size * 2), 1)
        self.linear2 = nn.Linear(18, 128)
        self.linear3 = nn.Linear(128, 1)

        #FeedForward
        self.fc = nn.Linear(12, 1)


    def forward(self, x):

        """
        #CNN
        embedded = self.embedding(x).permute(0,2,1)

        cnn_output = self.Conv1(embedded)
        cnn_output = self.dropout1(cnn_output)
        cnn_output = self.maxpool1(cnn_output)
        cnn_output = self.Conv2(cnn_output)
        cnn_output = self.dropout2(cnn_output)
        cnn_output = self.maxpool1(cnn_output)
        cnn_output = self.Conv3(cnn_output)
        cnn_output = self.dropout3(cnn_output).squeeze(1)
        """

        #LSTM
        embedded = self.embedding(x)
        
        h0 = torch.zeros(2 * self.num_layers, embedded.shape[0], self.hidden_size)
        c0 = torch.zeros(2 * self.num_layers, embedded.shape[0], self.hidden_size)

        lstm_output, _ = self.lstm(embedded, (h0, c0))
        lstm_output = torch.tanh(self.linear1(lstm_output))
        lstm_output = lstm_output.squeeze(2)
        lstm_output = torch.tanh(self.linear2(lstm_output))
        lstm_output = torch.tanh(self.linear3(lstm_output))

        

        #FeedForward
        #connected = torch.cat([lstm_output, cnn_output], dim = 1)
        #output = torch.tanh(self.fc(connected))

        return lstm_output.squeeze(1)
    




#new combined network
#first channel is local features using CNN
#second channel is global features using LSTM
class CNN_LSTM(nn.Module):

    def __init__(self, vocab_size = 21, embedding_size = 24, hidden_size = 64, num_layers = 2):
        super(CNN_LSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx = 0)

        #CNN
        self.conv1 = nn.Conv1d(in_channels = 18, out_channels = 128, kernel_size = 3, stride = 1)
        self.dropout1 = nn.Dropout()
        self.maxpool1 = nn.MaxPool1d(kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv1d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1)
        self.dropout2 = nn.Dropout()
        self.maxpool2 = nn.MaxPool1d(kernel_size = 3, stride = 1)
        self.cnnlinear1 = nn.Linear(16, 1)
        self.cnnlinear2 = nn.Linear(64, 1)

        #LSTM
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first = True, dropout = 0.2, bidirectional = True)
        self.lstmlinear1 = nn.Linear(int(self.hidden_size * 2), 1)
        self.lstmlinear2 = nn.Linear(18, 128)
        self.lstmlinear3 = nn.Linear(128, 1)



        #FeedForward
        self.fc1 = nn.Linear(82, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        
        embedded = self.embedding(x)

        #CNN
        output_cnn = self.conv1(embedded)
        output_cnn = self.dropout1(output_cnn)
        output_cnn = self.maxpool1(output_cnn)
        output_cnn = self.conv2(output_cnn)
        output_cnn = self.dropout2(output_cnn)
        output_cnn = self.maxpool2(output_cnn)
        output_cnn = torch.tanh(self.cnnlinear1(output_cnn)).squeeze()
        #output_cnn = torch.tanh(self.cnnlinear2(output_cnn))

        #LSTM

        h0 = torch.zeros(2 * self.num_layers, embedded.shape[0], self.hidden_size)
        c0 = torch.zeros(2 * self.num_layers, embedded.shape[0], self.hidden_size)

        output_lstm, _ = self.lstm(embedded, (h0, c0))
        output_lstm = torch.tanh(self.lstmlinear1(output_lstm))
        output_lstm = output_lstm.squeeze(2)
        #output_lstm = torch.tanh(self.lstmlinear2(output_lstm))
        #output_lstm = torch.tanh(self.lstmlinear3(output_lstm))

        #FeedForward
        combined_output = torch.cat([output_cnn, output_lstm], dim = 1)
        output = torch.tanh(self.fc1(combined_output))
        output = torch.tanh(self.fc2(output))

        return output.squeeze(1)
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(21, 24)
        

        self.conv1 = nn.Conv1d(in_channels = 18, out_channels = 128, kernel_size = 3, stride = 1)
        self.dropout1 = nn.Dropout()
        self.maxpool1 = nn.MaxPool1d(kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv1d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1)
        self.dropout2 = nn.Dropout()
        self.maxpool2 = nn.MaxPool1d(kernel_size = 3, stride = 1)
        self.fc1 = nn.Linear(16, 1)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):


        embedded = self.embedding(x)
        output = self.conv1(embedded)
        output = self.dropout1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.dropout2(output)
        output = self.maxpool2(output)
        output = torch.tanh(self.fc1(output)).squeeze()
        output = torch.tanh(self.fc2(output))

        return output.squeeze()

