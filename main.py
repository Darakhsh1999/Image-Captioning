import Params
from model import ImageCaptionGenerator


if __name__ == "__main__":
    p = Params.Params()
    model = ImageCaptionGenerator()
    model.train(p.n_epochs)