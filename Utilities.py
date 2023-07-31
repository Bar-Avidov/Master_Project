
import torch

a = torch.tensor([1.0,2.0,7.0]).float()
b = torch.tensor([1.0,2.0,7.2]).float()

loss_fc = torch.nn.MSELoss(reduction = 'none')

loss = loss_fc(a,b)
print(loss)