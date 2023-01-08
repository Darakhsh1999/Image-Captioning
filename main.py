import torch
from torch.utils.data import DataLoader

from Params import Params
from model import ImageCaptionGenerator
from model import train_ICG
import os
from flickerdata import FlickerData
import pickle

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

    vocab_lc = train_dataset.vocab_lc # vocabulary
    par = Params(vocab_size=len(vocab_lc)) # parameters

    # Dataloaders
    train_dataloader = DataLoader(dataset= train_dataset, batch_size= par.batch_size)
    val_dataloader = DataLoader(dataset= dev_dataset, batch_size= par.batch_size)
    test_dataloader = DataLoader(dataset= test_dataset, batch_size= par.batch_size)

    model = ImageCaptionGenerator(vocab_lc, par)
    model.to(par.device)
    if os.path.exists("Models/icg.pt"):
        model.load_state_dict(torch.load("Models/icg.pt"))

    if train_model:
        train_ICG(model, par, train_dataloader, val_dataloader)

    # Predict on test sentence
    model.to("cpu")
    model.eval()
    images, lemma, _ = dev_dataset[0]
    image_input = torch.unsqueeze(images, dim=0) 
    print(image_input.shape)
    model_out = model.predict(image_input)
    print(len(model_out[0]))
    print(f"model prediction{dev_dataset.decode_lc(model_out[0])}")
    print("Target caption:", dev_dataset.decode_lc(lemma))



