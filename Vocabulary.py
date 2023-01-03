""" Called by the image caption generator when we want to create a vocabulary from the train captions """
import spacy

nlp = spacy.load("en_core_web_sm")
from collections import Counter


class Vocabulary():

    def __init__(self, data, max_vocab_len=None):
        """ Creates a vocabulary using training data.
                            Special symbols, <BOS>,<EOS>,<PAD>,<UNK> """

        self.vocab = {}
        self.s_vocab = len(self.max_vocab_len)

        UNKNOWN = "<UNKNOWN>"
        BOS = "<BOS>"
        EOS = "<EOS>"
        PAD = "<PAD>"

        lemma_captions = []
        for i in range(self.n_images):
            lemma_captions.append(self.data[i][1])  # append lemma caption

        word_count = Counter(t for x in lemma_captions for t in [t.text.lower() for t in nlp.tokenizer(x)])

        self.vocab[UNKNOWN] = 0
        self.vocab[BOS] = 1
        self.vocab[EOS] = 2
        self.vocab[PAD] = 3

        if max_vocab_len is not None:
            word_list = word_count.most_common(max_vocab_len - 4)
        else:
            word_list = word_count

        for i, (w, count) in enumerate(word_list):
            self.vocab[w] = i + 4

    def get_vocab(self):
        return self.vocab
