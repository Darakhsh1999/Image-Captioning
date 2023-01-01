import torch
import torch.nn as nn
import torchvision
import Params

class Encoder(torch.nn.Module):

    def __init__(self, p):
        super().__init__()
        
        #  load in encoder
        self.vgg = torchvision.models.vgg19()
        for _, param in self.vgg.named_parameters(): # freeze vgg weights
            param.requires_grad = False

        self.latent_embedding = nn.Linear(4096, p.emb_dim) # maps latent vector to word embedding
        
    def forward(self, X):
        """ Takes in image X and outputs latent vector """

        batch_size = X.shape[0]

        X = self.vgg.features(X) # convolutional part
        X = self.vgg.avgpool(X) # sets to fixed size (512,7,7)
        X = X.view(batch_size, 512*7*7) # flatten
        X = self.vgg.classifier[0:3](X) 
        X = self.latent_embedding(X)
        return X


class Decoder(torch.nn.Module):
    """ Decodes a latent vector into lemmas/captions """

    def __init__(self, p):
        super().__init__()

        self.p = p
        self.lstm = nn.LSTM(
            input_size= p.emb_dim,
            hidden_size= p.hidden_size,
            num_layers= p.n_layers,
            batch_first= True)
        self.word_emb = nn.Embedding(p.emb_dim, p.vocab_size) 
        self.output_emb = nn.Linear(p.hidden_size, p.vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, latent_vector, caption):

        inputs = torch.cat((latent_vector, caption), dim= 0)
        output, _ = self.lstm(inputs)
        logits = self.output_emb(output)
        p_word = self.softmax(logits)
        return p_word

    def predict(self, latent_embedding):
        """ Used to prediction on images during testing """
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

    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.loss = None
        self.optim = None
        pass

    def forward(self, images, captions):
        pass

    def predict(self, image):
        """ Takes in image and returns caption """
        latent_image = self.encoder(image)
        return self.decoder.predict(latent_image)

        


if __name__ == "__main__":
        
    batch_size = 8
    p = Params.Params(vocab_size= 100)

    # Encoder testing
    encoder = Encoder(p)
    A = torch.rand(batch_size,3,250,300) # (batch, channel, width, height) 
    A_out = encoder(A)
    print(A_out.shape)

    # Decoder testing
    decoder = Decoder(p)
    B_out = decoder(A_out)
