""" Called by the image caption generator when we want to create a vocabulary from the train captions """


class Vocabulary():

    def __init__(self, max_vocab_len = None):
        self.vocab = []
        self.s_vocab = len(self.vocab)

    def make_vocab(self, max_vocab_len):
        pass