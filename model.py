import torch
import torch.nn as nn
import torchvision

class Encoder(torch.nn.Module):

    def __init__(self, p):
        super().__init__()
        
        #  load in encoder
        self.vgg = torchvision.models.vgg19()
        for param in self.vgg.named_parameters(): # freeze vgg weights
            param.requires_grad = False

        self.latent_embedding = nn.Linear(4096, p.emb_dim) # only trainable layer in encoder
        
    def forward(self, X):
        """ Takes in image X and outputs latent vector """

        batch_size = X.shape[0]

        X = self.vgg.features(X) # convolutional part
        X = self.vgg.avgpool(X) # sets to fixed size (512,7,7)
        X = X.view(batch_size, 512*7*7) # flatten
        X = self.vgg.classifier[0:3](X) 
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
        self.word_emb = nn.Embedding() # TODO fill
        self.output_emb = nn.Linear(p.hidden_size, p.vocab_size)

    def forward(self, latent_embedding, mode= "train"):
        pass
        
        if mode == "train":
            lstm_output, _ = self.lstm() # (batch_size, seq_len, hidden_size)

        elif mode == "test": # autogenerative

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
                if (next_token is "<EOS>"):
                    break

            return sentence 

        else:
            raise ValueError(f"Model only has train and test modes, but recieved {mode}")

class ImageCaptionGenerator(torch.nn.Module):

    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.loss = None
        self.optim = None
        pass

    def forward(self):
        pass

    def train(self):
        pass

if __name__ == "__main__":
        

    pretrained_model = Encoder()

    batch_size = 5
    A = torch.rand(batch_size,3,224,224)
    A = pretrained_model(A)
    print(A.shape)
