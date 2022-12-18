import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time


vgg_transform = torchvision.models.VGG19_Weights.IMAGENET1K_V1.transforms()

class FlickerData(Dataset):

    def __init__(self, subset):

        if subset.lower() not in ["train","dev","test"]:
            raise ValueError("Not valid option.")

        image_path = f"Data\Flicker8k_text\Flickr_8k.{subset.lower()}Images.txt" 
        lemma_caption_path = f"Data\Flicker8k_text\Flickr8k.lemma.token.txt" 
        caption_path = f"Data\Flicker8k_text\Flickr8k.token.txt" 

        # Load images
        with open(image_path) as f:
            image_handles = f.read().splitlines() # list of image IDs
        self.n_images = len(image_handles)
        self.images = torch.zeros((self.n_images, 3, 224, 224)) # (width, height) -(resize)> (256,256) -(central crop)> (224,244)
        self.labels = [] # (lemma caption, token caption)

        # Load in labels
        lemma_caption_df = pd.read_csv(lemma_caption_path, sep= "\t", names= ["captionID", "caption"])
        caption_df = pd.read_csv(caption_path, sep= "\t", names= ["captionID", "caption"])

        # Transform images from PIL to torch tensor & search for labels
        for img_idx, image_file in enumerate(image_handles):
            
            # Images
            #pil_image = Image.open(image_file).resize((256,256), resample= Image.BILINEAR)
            pil_image = Image.open(f"Data\Flicker8k_images\{image_file}")
            transformed_image = vgg_transform(pil_image)
            self.images[img_idx,:,:,:] = transformed_image #(n_images, n_channels, width, height)

            # Labels
            lemma_caption_frame = lemma_caption_df.loc[lemma_caption_df["captionID"] == (image_file+"#0") ]
            lemma_caption = lemma_caption_frame.iloc[0]["caption"]

            caption_frame = caption_df.loc[caption_df["captionID"] == (image_file+"#0") ]
            caption = caption_frame.iloc[0]["caption"]

            self.labels.append((lemma_caption, caption))


    def __getitem__(self, index):
        return (self.images[index,:,:,:], self.labels[index][0], self.labels[index][1]) # (torch.tensor, str, str)

    def __len__(self):
        return self.n_images
        

if __name__ == "__main__":
    print("hello")
    t0 = time.time()
    loader_data = DataLoader(dataset= FlickerData(subset= "train"), batch_size= 3) 
    t_end = time.time() -t0
    print(f"Data loading time {t_end:.3f} s")
    iter_loader = iter(loader_data)
    images, lemmas, captions = next(iter_loader)
    print(images.shape)
    print(lemmas)
    print(type(lemmas))
    print(captions)
    print(type(captions))