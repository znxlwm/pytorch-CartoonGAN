from torch import nn
from torch.nn.functional import l1_loss
from torchvision.models import vgg19


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True)
        del self.vgg.features[26:]
        del self.vgg.classifier
        del self.vgg.avgpool

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, y_hat, y):
        self.vgg.eval()  # do not track batch statistics etc.
        return l1_loss(self.vgg.features(y_hat), self.vgg.features(y))
