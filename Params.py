import torch.nn as nn

class Params():

    def __init__(self, vocab_size):
        vocab_size = vocab_size


    # Architecture
    emb_dim = 512 # same as Google paper
    batch_size = 10
    max_sen_len = 20
    hidden_size = 256
    n_layers = 2

    # Training
    n_epochs = 50
    lr = 0.001
    
