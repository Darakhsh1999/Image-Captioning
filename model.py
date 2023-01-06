import torch
import torch.nn as nn
import torchvision
import pickle
import os
from tqdm import trange
from torch.utils.data import DataLoader
from Params import Params
from flickerdata import FlickerData


class Encoder(torch.nn.Module):
    """ Encodes image into an embeding vector used for decoder """

    def __init__(self, p):
        super().__init__()

        self.vgg = torchvision.models.vgg19()  # load in encoder
        for _, param in self.vgg.named_parameters():  # freeze vgg weights
            param.requires_grad = False

        self.latent_embedding = nn.Linear(4096, p.emb_dim)  # maps latent_vector to embed_vector

    def forward(self, X):
        """ Takes in image X and outputs embed_vector """

        batch_size = X.shape[0]

        X = self.vgg.features(X)  # convolutional part
        X = self.vgg.avgpool(X)  # sets to fixed size (512,7,7)
        X = X.view(batch_size, 512 * 7 * 7)  # (batch,25088) flatten
        X = self.vgg.classifier[0:3](X)  # (batch,4096)
        X = self.latent_embedding(X)  # (batch,emb_dim)
        return X


class Decoder(torch.nn.Module):
    """ Decodes a embed_vector into lemmas/captions """

    def __init__(self, p):
        super().__init__()

        self.p = p
        self.lstm = nn.LSTM(
            input_size=p.emb_dim,
            hidden_size=p.hidden_size,
            num_layers=p.n_layers,
            batch_first=True)
        self.word_emb = nn.Embedding(p.vocab_size, p.emb_dim)
        self.output_emb = nn.Linear(p.hidden_size, p.vocab_size)

    def forward(self, embed_vector, caption):
        """ Takes in embed_vector and captions and outputs prob distribution 
            Input:
                embed_vector (batch, emb_dim)
                caption (batch, max_sen_len)
            Output:
                probability (batch, max_sen_len, vocab_size)
        """

        batch_size = embed_vector.shape[0]
        embed_captions = self.word_emb(caption)  # (batch_size, max_sen_len, emb_dim)
        inputs = torch.cat(
            (embed_vector.view(batch_size, 1, self.p.emb_dim), embed_captions),
            dim=1)  # (batch_size, max_sen_len+1, emb_dim)
        output, _ = self.lstm(inputs)  # (batch_size, max_sen_len+1, hidden_size)
        logits = self.output_emb(output)  # (batch_size, max_sen_len+1, vocab_size)
        return logits

    def predict(self, latent_embedding):
        """ Used to predict caption for images during testing """
        sentence = []
        lstm_out, _ = self.lstm(latent_embedding)  # (batch, emb_dim) -> (batch_size, seq_len, hidden_size)
        prob = self.output_emb(lstm_out)
        norm_prob = nn.functional.softmax(prob)

        # sample word (greedy)

        next_token = norm_prob.argmax()
        sentence.append(next_token)

        sentence_length = 1
        while (sentence_length <= self.p.max_sen_len):
            emb_token = self.word_emb(next_token)  # integer-representation -> word_embedding
            lstm_out, _ = self.lstm(torch.unsqueeze(emb_token, dim=0))
            prob = self.output_emb(lstm_out)
            norm_prob = nn.functional.softmax(prob)
            next_token = norm_prob.argmax()
            sentence.append(next_token)
            sentence_length += 1
            if (next_token.item() == 3):  # "<EOS>" change to vocab.get
                break

        return sentence


class ImageCaptionGenerator(torch.nn.Module):

    def __init__(self, p: Params):

        super().__init__()

        # Load in data and make dataloaders
        if os.path.exists("Datasets/dataset.p"):
            train_dataset: FlickerData
            dev_dataset: FlickerData
            test_dataset: FlickerData
            with open("Datasets/dataset.p", "rb") as f:
                train_dataset, dev_dataset, test_dataset = pickle.load(f)
        else:
            raise FileExistsError("Run flickerdata.py first to create dataset")
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=p.batch_size)
        self.val_dataloader = DataLoader(dataset=dev_dataset, batch_size=p.batch_size)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=p.batch_size)

        # Create vocabulary
        self.vocab_lc = train_dataset.vocab_lc
        self.vocab_c = train_dataset.vocab_c

        self.p = p
        self.encoder = Encoder(p)
        self.decoder = Decoder(p)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_lc.get("<PAD>"))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=p.lr)  # ADAMS ?
        pass

    def forward(self, image, lemma):
        """ Takes in images and returns probability distribution for each token """
        latent_embedding = self.encoder(image)
        return self.decoder(latent_embedding, lemma)

    def predict(self, image):
        """ Takes in image and returns caption """
        latent_image = self.encoder(image)
        return self.decoder.predict(latent_image)


def train_ICG(model: ImageCaptionGenerator, train_dataloader, dev_dataloader, n_epochs):
    PAD = model.vocab_lc["<PAD>"]
    # define loss and optimzer here
    # define device

    for epoch in trange(n_epochs):

        for batch_idx, batch in enumerate(train_dataloader):
            # images: [batch,channel,width,height], lemmas: [batch,max_sen_len]
            images, lemmas, _ = batch
            print(torch.tensor(PAD).repeat(10, 1))
            target = torch.cat((lemmas, torch.tensor(PAD).repeat(10, 1)), dim=1)
            print(f"target is of shape {target.shape}")
            target = torch.flatten(target)
            print(f"flattened target is of shape {target.shape}")

            # forward pass
            model.train()
            output_logits = model(images, lemmas)
            print(f"output from model is of shape {output_logits.shape}")
            output_logits = torch.flatten(output_logits, start_dim=0, end_dim=1)
            print(f"flattened output from model is of shape {output_logits.shape}")
            # softmax ? 

            #  calculate loss
            model.optimizer.zero_grad()

            loss = model.criterion(output_logits, target)
            print(loss.item())
            loss.backward()

            # optimizer step
            model.optimizer.step()

            # model.eval()
            # validation stuff (ROGUE/BLEU) , use prediuct function


if __name__ == "__main__":
    batch_size = 8
    p = Params.Params(vocab_size=100)

    # Encoder testing
    encoder = Encoder(p)
    A = torch.rand(batch_size, 3, 250, 300)  # (batch, channel, width, height)
    A_out = encoder(A)
    print(A_out.shape)

    # Decoder testing
    # decoder = Decoder(p)
    # B_out = decoder(A_out)
