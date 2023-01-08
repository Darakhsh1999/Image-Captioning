import time
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from Params import Params
from collections import defaultdict


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

    def predict(self, latent_embedding, vocab):
        """ Used to predict caption for images during testing 
            
            Input:
              latent_embedding (batch_size, emb_dim) -> batch_size number of sentences
            Output:
        """
        n_sentences = latent_embedding.shape[0]
        sentences = []

        for sen_idx in range(n_sentences):

            sentence = []
            
            # first token
            print("input size", latent_embedding[sen_idx,:].shape)
            lstm_out, (h_t, c_t) = self.lstm(latent_embedding[sen_idx,:], (0,0)) # (hidden_size,)
            print("LSTM output shape", lstm_out.shape)
            prob = self.output_emb(lstm_out)
            print("probabilities shape", prob.shape)
            norm_prob = nn.functional.softmax(prob)
            print("normalized probabilities shape", norm_prob.shape)


            next_token = norm_prob.argmax()
            sentence.append(next_token)

            # while loop until EOS or max_sen_len
            while len(sentence) < p.max_pred_sen:

                lstm_out, (h_t, c_t) = self.lstm(latent_embedding[sen_idx,:], (h_t, c_t)) 
                prob = self.output_emb(lstm_out)
                norm_prob = nn.functional.softmax(prob)
                next_token = norm_prob.argmax()
                sentence.append(next_token)

                if next_token == vocab.get("<EOS>"):
                    break

            sentences.append(sentence)

        return sentences


class ImageCaptionGenerator(torch.nn.Module):

    def __init__(self, vocab_lc, par: Params):
        super().__init__()

        # parameters and vocab
        self.vocab_lc = vocab_lc
        self.p = par

        # set up network layers
        self.encoder = Encoder(par)
        self.decoder = Decoder(par)

    def forward(self, image, lemma):
        """ Takes in images and returns probability distribution for each token """
        latent_embedding = self.encoder(image)
        return self.decoder(latent_embedding, lemma)

    def predict(self, image):
        """ Takes in image and returns caption """
        latent_image = self.encoder(image)
        return self.decoder.predict(latent_image, self.vocab_lc)


def train_ICG(model: ImageCaptionGenerator, par: Params, train_dataloader, dev_dataloader):
    
    device = par.device

    # Define loss and optimizer
    PAD = model.vocab_lc["<PAD>"]
    loss_func = nn.CrossEntropyLoss(ignore_index= PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr= par.lr)

    # Contains the statistics that will be returned.
    history = defaultdict(list)

    progress = tqdm(range(par.n_epochs), 'Epochs')
    print(f" Started training for {par.n_epochs} epoch, n_batches = {len(train_dataloader)}, using device: {par.device}")
    for epoch in progress:

        t0 = time.time()

        # run one epoch of training
        model.train()
        training_loss = 0
        num_batches = len(train_dataloader)
        for batch_idx, batch in enumerate(train_dataloader):

            # images: [batch,channel,width,height], lemmas: [batch,max_sen_len]
            images, lemmas, _ = batch
            images = images.to(device)
            lemmas = lemmas.to(device) # (batch_size, max_sen_len)

            # set target to padded lemmas
            target = torch.cat((lemmas, torch.tensor(PAD).repeat(par.batch_size, 1).to(par.device)), dim=1) # (batch_size, max_sen_len+1)
            target = torch.flatten(target)  # flatten batch dims (batch_size*(max_sen_len+1),)

            # forward pass
            output_logits = model(images, lemmas) # (batch_size, max_sen_len+1, vocab_size)
            output_logits = torch.flatten(output_logits, start_dim=0, end_dim=1)  # (batch_size*(max_sen_len+1), vocab_size)

            # calculate loss
            loss = loss_func(output_logits, target)
            training_loss += loss.item()

            # update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_loss /= len(train_dataloader)
        print(f"batch training loss: {training_loss}")

        # run one epoch of validation
        model.eval()
        validation_loss = 0  # "Not implemented"
        validation_acc = 0  # "Not implemented" - rouge/bleu
        '''with torch.no_grad():
            for batch_idx, batch in enumerate(dev_dataloader):
                # images: [batch,channel,width,height], lemmas: [batch,max_sen_len]
                images, lemmas, _ = batch
                images = images.to(device)
                lemmas = lemmas.to(device)

                # forward pass
                predicted_tokens_encoded = model.predict(images)

                # calculate loss
                pass

                # calculate accuracy or rouge/bleu-scores
                pass'''

        t1 = time.time()

        # Save epoch data
        history['train_loss'].append(training_loss)
        history['val_loss'].append(validation_loss)
        history['val_acc'].append(validation_acc)
        history['time'].append(t1 - t0)

        progress.set_postfix({
            'time': f'{t1 - t0:.2f}',
            'train_loss': f'{training_loss:.2f}',
            'val_loss': f'{validation_loss:.2f}',
            'val_acc': f'{validation_acc:.2f}'})
        torch.save(model.state_dict(), "Models/icg.pt")  # better to save model that performed best on validation set


if __name__ == "__main__":
    batch_size = 8
    p = Params(vocab_size=100)

    # Encoder testing
    encoder = Encoder(p)
    A = torch.rand(batch_size, 3, 250, 300)  # (batch, channel, width, height)
    A_out = encoder(A)
    print(A_out.shape)

    # Decoder testing
    # decoder = Decoder(p)
    # B_out = decoder(A_out)
