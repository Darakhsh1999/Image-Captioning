import Params
from model import ImageCaptionGenerator
from model import train_ICG


if __name__ == "__main__":
    p = Params.Params()
    model = ImageCaptionGenerator()
    train_ICG(model,  n_epochs= 10)