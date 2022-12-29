""" Defines the Flickr dataset """
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time

vgg_transform = torchvision.models.VGG19_Weights.IMAGENET1K_V1.transforms()


class FlickerData(Dataset):

    def __init__(self, subset):

        if subset.lower() not in ["train", "dev", "test"]:
            raise ValueError("Not valid option.")

        image_path = f"Data\Flicker8k_text\Flickr_8k.{subset.lower()}Images.txt"
        lemma_caption_path = f"Data\Flicker8k_text\Flickr8k.lemma.token.txt"
        caption_path = f"Data\Flicker8k_text\Flickr8k.token.txt"

        # Load images
        with open(image_path) as f:
            image_handles = f.read().splitlines()  # list of image IDs
        self.n_images = len(image_handles)
        self.images = torch.zeros(
            (self.n_images, 3, 224, 224))  # (width, height) -(resize)> (256,256) -(central crop)> (224,224)
        self.labels = []  # (lemma caption, token caption)

        # Load in labels
        df_lc = pd.read_csv(lemma_caption_path, sep="\t", names=["captionID", "caption"])
        df_c = pd.read_csv(caption_path, sep="\t", names=["captionID", "caption"])

        df_lemma_caption = pd.DataFrame()
        df_lemma_caption[['image', 'id']] = df_lc.pop('captionID').str.split('#', n=1, expand=True)
        df_lemma_caption = df_lemma_caption.join(df_lc.pop('caption'))
        df_lemma_caption = df_lemma_caption[df_lemma_caption['id'] == '0']
        df_lemma_caption.pop('id')
        image_to_lemma_caption = dict(df_lemma_caption.values)  # {'image_file': 'lemma caption'}

        df_caption = pd.DataFrame()
        df_caption[['image', 'id']] = df_c.pop('captionID').str.split('#', n=1, expand=True)
        df_caption = df_caption.join(df_c.pop('caption'))
        df_caption = df_caption[df_caption['id'] == '0']
        df_caption.pop('id')
        image_to_caption = dict(df_caption.values)  # {'image_file': 'caption'}

        # Transform images from PIL to torch tensor & search for labels
        for img_idx, image_file in enumerate(image_handles):

            # Images
            pil_image = Image.open(f"Data\Flicker8k_images\{image_file}")
            transformed_image = vgg_transform(pil_image)
            self.images[img_idx, :, :, :] = transformed_image  # (n_images, n_channels, width, height)

            # Labels
            lemma_caption = image_to_lemma_caption[image_file]
            caption = image_to_caption[image_file]

            self.labels.append((lemma_caption, caption))

    def __getitem__(self, index):
        """ Returns tuple (image, lemma_caption, caption) """
        return (self.images[index, :, :, :], self.labels[index][0], self.labels[index][1])  # (torch.tensor, str, str)

    def __len__(self):
        return self.n_images


if __name__ == "__main__":

    t0 = time.time()
    loader_data = DataLoader(dataset=FlickerData(subset="train"), batch_size=3) # dataloader
    t_end = time.time() - t0
    print(f"Data loading time {t_end:.3f} s")
    iter_loader = iter(loader_data)
    images, lemmas, captions = next(iter_loader) # images: tensor, lemmas: tuple, captions: tuple
    print(images.shape) # batch_size, n_channels, width, height
    print(lemmas)
    print(captions)