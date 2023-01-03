import torch
import torch.nn as nn
import torchvision
import Params
import spacy

nlp = spacy.load("en_core_web_sm")
from flickerdata import FlickerData
from collections import Counter


class Encoder(torch.nn.Module):
    """ Encodes image into an embeding vector used for decoder """

    def __init__(self, p):
        super().__init__()

        self.vgg = torchvision.models.vgg19() #  load in encoder
        for _, param in self.vgg.named_parameters(): # freeze vgg weights
            param.requires_grad = False

        self.latent_embedding = nn.Linear(4096, p.emb_dim) # maps latent_vector to embed_vector
        
    def forward(self, X):
        """ Takes in image X and outputs embed_vector """

        batch_size = X.shape[0]

        X = self.vgg.features(X) # convolutional part
        X = self.vgg.avgpool(X) # sets to fixed size (512,7,7) 
        X = X.view(batch_size, 512*7*7) # (batch,25088) flatten
        X = self.vgg.classifier[0:3](X) # (batch,4096)
        X = self.latent_embedding(X) # (batch,emb_dim)
        return X


class Decoder(torch.nn.Module):
    """ Decodes a embed_vector into lemmas/captions """

    def __init__(self, p):
        super().__init__()

        self.p = p
        self.lstm = nn.LSTM(
            input_size= p.emb_dim,
            hidden_size= p.hidden_size,
            num_layers= p.n_layers,
            batch_first= True)
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
        embed_captions = self.word_emb(caption) # (batch_size, max_sen_len, emb_dim)
        inputs = torch.cat( 
            (embed_vector.view(batch_size, 1, self.p.emb_dim), embed_captions),
            dim= 1) # (batch_size, max_sen_len+1, emb_dim)
        output, _ = self.lstm(inputs) # (batch_size, max_sen_len+1, hidden_size)
        logits = self.output_emb(output) # (batch_size, max_sen_len+1, vocab_size)
        return logits

    def predict(self, latent_embedding):
        """ Used to predict caption for images during testing """
        sentence = []
        lstm_out, _ = self.lstm(latent_embedding) # (batch, emb_dim) -> (batch_size, seq_len, hidden_size)
        prob = self.output_emb(lstm_out)
        norm_prob = nn.functional.softmax(prob) 

        # sample word (greedy)

        next_token = norm_prob.argmax()
        sentence.append(next_token)
        
        sentence_length = 1
        while (sentence_length <= self.p.max_sen_len):
            emb_token = self.word_emb(next_token) # integer-representation -> word_embedding
            lstm_out, _ = self.lstm(emb_token) 
            prob = self.output_emb(lstm_out)
            norm_prob = nn.functional.softmax(prob) 
            next_token = norm_prob.argmax()
            sentence.append(next_token)
            sentence_length += 1
            if (next_token == "<EOS>"):
                break

        return sentence 


class ImageCaptionGenerator(torch.nn.Module):

    def __init__(self, mode, p):
        if mode not in ["train", "dev" "test"]:
            raise ValueError(f"Incorrect mode {mode}")
        self.vocab = self.make_vocab(FlickerData(mode))

        self.encoder = Encoder(p)
        self.decoder = Decoder(p)
        self.loss = None # CE
        self.optim = None # ADAMS ? 
        pass

    def make_vocab(self, data):
        """ Creates a vocabulary using training data.
            Special symbols, <BOS>,<EOS>,<PAD>,<UNK> """

        UNKNOWN = "<UNKNOWN>"
        BOS = "<BOS>"
        EOS = "<EOS>"

        lemma_captions = []
        for i in range(self.n_images):
            lemma_captions.append(self.data[i][1])  # append lemma caption

        freqs = Counter(t for x in lemma_captions for t in [t.text.lower() for t in nlp.tokenizer(x)])

        vocab = {}
        vocab[UNKNOWN] = 0
        vocab[BOS] = 1
        vocab[EOS] = 2

        freq_list = freqs

        for i, (w, count) in enumerate(freq_list):
            self.vocab_x[w] = i + 3

        return vocab

    def forward(self, image):
        """ Takes in images and returns probability distribution for each token """
        latent_embedding = self.encoder(image)
        return self.decoder(latent_embedding)
        
    def predict(self, image):
        """ Takes in image and returns caption """
        latent_image = self.encoder(image)
        return self.decoder.predict(latent_image)

    def train(self, n_epochs):
        pass

if __name__ == "__main__":
        
    batch_size = 8
    p = Params.Params(vocab_size= 100)

    # Encoder testing
    encoder = Encoder(p)
    A = torch.rand(batch_size,3,250,300) # (batch, channel, width, height) 
    A_out = encoder(A)
    print(A_out.shape)

    # Decoder testing
    #decoder = Decoder(p)
    #B_out = decoder(A_out)
