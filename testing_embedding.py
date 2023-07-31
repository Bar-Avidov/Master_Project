import torch
from transformers import BertTokenizer, BertModel, pipeline
import pandas as pd
import data_manager

data = pd.read_excel("~/Project/Neural-Network/data_from_Ens_Grad.xlsx")
data = data.dropna().reset_index(drop = True)

train_x, train_y, test_x, test_y, enrichment_factor = data_manager.data_manager_for_LSTM(data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Rostlab/prot_bert_bfd"
tokenizer_name = "Rostlab/prot_bert_bfd"
#tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
#model = BertModel.from_pretrained("Rostlab/prot_bert_bfd").to(device)
list = []
embedding_pipeline = pipeline(task ="feature-extraction", model = model_name, tokenizer = tokenizer_name, device = 0)

embeddings = []
for i in range(len(train_x)):
    print(i)

    sequence = " ".join(train_x[i])
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    results = torch.tensor(embedding_pipeline(sequence))
    embeddings.append(results)

    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    torch.cuda.empty_cache()