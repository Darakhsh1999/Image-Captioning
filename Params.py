import torch

class Params:

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Architecture
    emb_dim = 512  # same as Google paper
    batch_size = 10
    max_sen_len = 20
    hidden_size = 256
    n_layers = 2

    # Training
    n_epochs = 10
    lr = 0.01
