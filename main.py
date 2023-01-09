import torch
import torch.nn as nn
import pickle
import os
from flickerdata import FlickerData
from torch.utils.data import DataLoader
from Params import Params
from model import ImageCaptionGenerator
from model import train_ICG, predict_one, performance_scores

if __name__ == "__main__":

    train_model = False

    # Load in preprocessed dataset
    if os.path.exists("Datasets/dataset.p"):
        train_dataset: FlickerData
        dev_dataset: FlickerData
        test_dataset: FlickerData
        with open("Datasets/dataset.p", "rb") as f:
            train_dataset, dev_dataset, test_dataset = pickle.load(f)
    else:
        raise FileExistsError("Run flickerdata.py first to create dataset")

    vocab_lc = train_dataset.vocab_lc # vocabulary from train data
    p = Params(vocab_size=len(vocab_lc)) # parameters

    # Dataloaders
    train_dataloader = DataLoader(dataset= train_dataset, batch_size= p.batch_size, shuffle= True)
    val_dataloader = DataLoader(dataset= dev_dataset, batch_size= p.batch_size, shuffle= False)
    test_dataloader = DataLoader(dataset= test_dataset, batch_size= p.batch_size, shuffle= False)

    # Model instance
    model = ImageCaptionGenerator(vocab_lc, p)
    model.to(p.device)
    if os.path.exists("Models/icg.pt"):
        model.load_state_dict(torch.load("Models/icg.pt"))

    # Loss and Optimizer
    PAD = vocab_lc["<PAD>"]
    loss_func = nn.CrossEntropyLoss(ignore_index= PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr= p.lr)

    if train_model:
        train_ICG(model, p, loss_func, optimizer, train_dataloader, val_dataloader)


    # Predict on test sentence
    prediction, target = predict_one(model, test_dataset)
    
    print(f"{'Prediction:':15} {prediction}") 
    print(f"{'Target:':15} {target}") 




    