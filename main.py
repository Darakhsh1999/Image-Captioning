import torch

import Params
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

    print(f"vocab size is {len(train_dataset.vocab_lc)}")
    p = Params.Params(vocab_size=len(train_dataset.vocab_lc))

    model = ImageCaptionGenerator(p)
    train_ICG(model, model.test_dataloader, model.val_dataloader, p)

    # check prediction
    model.eval()
    images, lemma, _ = dev_dataset[0]
    print(f"model prediction{dev_dataset.decode_lc(model.predict(torch.unsqueeze(images, dim=0)))}")
    print(dev_dataset.decode_lc(lemma))



