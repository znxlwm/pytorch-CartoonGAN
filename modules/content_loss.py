from torch import nn
from torchvision.models import vgg19


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True)

        # del self.vgg.features[xx:]

    def forward(self, y_hat, y):
        self.vgg(y_hat)