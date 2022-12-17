import torch
import torchvision

class Encoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        #  load in encoder
        self.vgg = torchvision.models.vgg19()
        
    def forward(self, X):

        batch_size = X.shape[0]

        X = self.vgg.features(X)
        X = self.vgg.avgpool(X)
        X = X.view(batch_size, 512*7*7)
        X = self.vgg.classifier[0:2](X)
        return X

if __name__ == "__main__":
    pretrained_model = Encoder()

    batch_size = 5
    A = torch.rand(batch_size,3,256,256)
    A = pretrained_model(A)
    print(A.shape)

