""" Defines the Flickr dataset """
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm")
import pandas as pd
import time

vgg_transform = torchvision.models.VGG19_Weights.IMAGENET1K_V1.transforms()


class FlickerData(Dataset):

    def __init__(self, subset, vocab=None, max_vocab_len=None):

        if subset.lower() not in ["train", "dev", "test"]:
            raise ValueError(f"{subset} is not a valid option.")

        self.vocab = vocab
        self.max_vocab_len = max_vocab_len

        # Load in paths
        image_path = f"Data\Flicker8k_text\Flickr_8k.{subset.lower()}Images.txt"
        lemma_caption_path = f"Data\Flicker8k_text\Flickr8k.lemma.token.txt"
        caption_path = f"Data\Flicker8k_text\Flickr8k.token.txt"

        # Load images
        with open(image_path) as f:
            image_handles = f.read().splitlines()  # list of image IDs
        self.n_images = len(image_handles)
        # (width, height) -(resize)> (256,256) -(central crop)> (224,224)
        self.images = torch.zeros((self.n_images, 3, 224, 224))  
        self.labels = []  # (lemma caption, token caption)

        # Load in labels 
        df_lc = pd.read_csv(lemma_caption_path, sep="\t", names=["captionID", "caption"])
        df_c = pd.read_csv(caption_path, sep="\t", names=["captionID", "caption"])

        # Lemma caption for all images
        df_lemma_caption = pd.DataFrame()
        df_lemma_caption[['image', 'id']] = df_lc.pop('captionID').str.split('#', n=1, expand=True)
        df_lemma_caption = df_lemma_caption.join(df_lc.pop('caption'))
        df_lemma_caption = df_lemma_caption[df_lemma_caption['id'] == '0']
        df_lemma_caption.pop('id')
        image_to_lemma_caption = dict(df_lemma_caption.values)  # {'image_file': 'lemma caption'}

        # Caption for all images
        df_caption = pd.DataFrame()
        df_caption[['image', 'id']] = df_c.pop('captionID').str.split('#', n=1, expand=True)
        df_caption = df_caption.join(df_c.pop('caption'))
        df_caption = df_caption[df_caption['id'] == '0']
        df_caption.pop('id')
        image_to_caption = dict(df_caption.values)  # {'image_file': 'caption'}

        # Create vocab for train data
        if vocab is None:
            vocab_lc, inv_vocab_lc = self.make_vocab(image_to_lemma_caption.values())
            vocab_c, inv_vocab_c = self.make_vocab(image_to_caption.values())

        tokenized_lemmas = []
        tokenized_captions = []
        max_sen_len_lc = 0
        max_sen_len_c = 0

        # Transform images from PIL to torch tensor & search for labels
        for img_idx, image_file in enumerate(image_handles):

            # Images
            pil_image = Image.open(f"Data\Flicker8k_images\{image_file}")
            transformed_image = vgg_transform(pil_image)
            self.images[img_idx, :, :, :] = transformed_image  # (n_images, n_channels, width, height)

            # String labels for image handle
            lemma_caption = image_to_lemma_caption[image_file]
            caption = image_to_caption[image_file]

            # Integer encode tokenized string labels
            tokens_lc = [vocab_lc.get(token.text.lower(), vocab_lc["<UNK>"]) for token in nlp.tokenizer(lemma_caption)] 
            tokens_c = [vocab_c.get(token.text.lower(), vocab_c["<UNK>"]) for token in nlp.tokenizer(caption)] 

            if len(tokens_lc) > max_sen_len_lc: max_sen_len_lc = len(tokens_lc) # update max sentence lenght
            if len(tokens_c) > max_sen_len_c: max_sen_len_c = len(tokens_c) # update max sentence lenght
            
            # Store integer encoded captions in list
            tokenized_lemmas.append(tokens_lc)
            tokenized_captions.append(tokens_c)

        
        # integer tensor
        self.labels_lc = torch.zeros((self.n_images, max_sen_len_lc)) 
        self.labels_c = torch.zeros((self.n_images, max_sen_len_c)) 
        for idx in range(self.n_images):
            len_lc = len(tokenized_lemmas[idx])
            len_c = len(tokenized_captions[idx])
            self.labels_lc[idx,:] = torch.tensor(tokenized_lemmas[idx] + (max_sen_len_lc-len_lc)*[vocab_lc["<PAD>"]])
            self.labels_c[idx,:] = torch.tensor(tokenized_captions[idx] + (max_sen_len_c-len_c)*[vocab_c["<PAD>"]])


    def __getitem__(self, index):
        """ Returns tuple (image, lemma_caption, caption) """
        # (dataset_size, img, caption)[i] -> (tensor(3,224,224), tensor(max_sen_len), tensor(max_sen_len))
        return (self.images[index, :, :, :], self.labels_lc[index,:], self.labels_c[index,:]) 

    def __len__(self):
        return self.n_images

    def make_vocab(self, data):
        """ Creates a vocabulary using training data.
            Special symbols, <BOS>,<EOS>,<PAD>,<UNK>
            Parameters:
              data: list of captions/lemmas """

        # Special symbols
        PAD = "<PAD>" # 0
        UNK = "<UNK>" # 1
        BOS = "<BOS>" # 2
        EOS = "<EOS>" # 3
        special_char = [PAD, UNK, BOS, EOS]

        # Create word counts for all captions
        word_count = Counter(t for x in data for t in [t.text.lower() for t in nlp.tokenizer(x)])

        # List of words including special characters
        if self.max_vocab_len is None:
            word_list = special_char + list(word_count.keys())
        else:
            most_common_word = word_count.most_common(self.max_vocab_len - len(special_char)) 
            word_list = special_char + [word[0] for word in most_common_word]

        # Vocab dictionary
        vocab = {token: int_id for int_id, token in enumerate(word_list)}
        inv_vocab = {int_id: token for int_id, token in enumerate(word_list)}

        return vocab, inv_vocab

    def decode(self, caption, inv_vocab: dict):
        return [inv_vocab[id] for id in caption]

if __name__ == "__main__":

    t0 = time.time()
    dataset =  FlickerData(subset= "train", max_vocab_len= None)
    loader_data = DataLoader(dataset=FlickerData(subset="train"), batch_size=3)  # dataloader
    t_end = time.time() - t0
    print(f"Data loading time {t_end:.3f} s")
    iter_loader = iter(loader_data)
    images, lemmas, captions = next(iter_loader)  # images: tensor, lemmas: tuple, captions: tuple
    print(images.shape)  # batch_size, n_channels, width, height
    print(lemmas)
    print(captions)
