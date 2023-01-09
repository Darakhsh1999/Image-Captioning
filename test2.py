import torch

qq = [1,2,4,6,3,1,5,7,3,1,412,2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)