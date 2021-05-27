import pytorch_lightning as pl
import torchvision
from torch.optim import Adam

from modules.content_loss import ContentLoss
from modules.generator import Generator


class PretrainGeneratorModule(pl.LightningModule):
    def __init__(self, lr_pretrain=0.002, b1=0.5, b2=0.999, in_nc=3, out_nc=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.content_loss = ContentLoss()
        self.generator = Generator(in_nc, out_nc)

    def training_step(self, batch, *args, **kwargs):
        g_out = self.generator(batch)

        sample_imgs = g_out[:4]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, 0)

        return self.content_loss(g_out, batch)

    def configure_optimizers(self):
        opt_g = Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        return opt_g

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Pretrain Generator")
        parser.add_argument('--lr_pretrain', type=float, default=0.002, help='learning rate pretraining')
        return parent_parser
