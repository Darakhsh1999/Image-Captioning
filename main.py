import torch
from torch.utils.data import DataLoader

from Params import Params
from model import ImageCaptionGenerator
from model import train_ICG
import os
from flickerdata import FlickerData
import pickle

if __name__ == "__main__":

    # Load in data and make dataloaders
    if os.path.exists("Datasets/dataset.p"):
        train_dataset: FlickerData
        dev_dataset: FlickerData
        test_dataset: FlickerData
        with open("Datasets/dataset.p", "rb") as f:
            train_dataset, dev_dataset, test_dataset = pickle.load(f)
    else:
        raise FileExistsError("Run flickerdata.py first to create dataset")

    vocab_lc = train_dataset.vocab_lc
    par = Params(vocab_size=len(vocab_lc))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=par.batch_size)
    val_dataloader = DataLoader(dataset=dev_dataset, batch_size=par.batch_size)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=par.batch_size)

    # create model and start training
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    model = ImageCaptionGenerator(vocab_lc, par)
    model.to(device)
    if os.path.exists("Models/icg.pt"):
        model.load_state_dict(torch.load("Models/icg.pt"))

    train_ICG(model, train_dataloader, val_dataloader, par)

    # check prediction
    model.eval()
    images, lemma, _ = dev_dataset[0]
    print(f"model prediction{dev_dataset.decode_lc(model.predict(torch.unsqueeze(images, dim=0)))}")
    print(dev_dataset.decode_lc(lemma))



