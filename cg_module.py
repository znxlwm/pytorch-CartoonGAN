import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from modules.content_loss import ContentLoss
from modules.discriminator import Discriminator
from modules.generator import Generator


def adversarial_loss(y_hat, y):
    return F.binary_cross_entropy(y_hat, y)


class CGModule(pl.LightningModule):
    def __init__(self,
                 pretrained_generator=None,
                 con_lambda=10.0,
                 lr_g=0.0002,
                 lr_d=0.0002,
                 in_nc=3,
                 out_nc=3,
                 b1=0.5,
                 b2=0.999,
                 *args,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters('con_lambda', 'out_nc', 'in_nc', 'lr_g', 'lr_d', 'b1', 'b2')

        self.generator = pretrained_generator if pretrained_generator is not None else Generator(in_nc, out_nc)
        self.discriminator = Discriminator(in_nc, 1)
        self.content_loss = ContentLoss()

    def forward(self, z):
        return self.generator(z)

    @staticmethod
    def set_requires_grad(model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def training_step(self, batch, batch_idx, optimizer_idx):
        src, tgt = batch
        tgt, edge = tgt[:, :3], tgt[:, 3:]
        real = torch.ones(src.size(0), 1, src.size(2) // 4, src.size(3) // 4).type_as(src)
        unreal = torch.zeros(src.size(0), 1, src.size(2) // 4, src.size(3) // 4).type_as(src)

        if optimizer_idx == 0:
            # train generator
            self.set_requires_grad(self.discriminator, False)
            g_out = self.generator(src)

            d_unreal = self.discriminator(g_out)
            d_unreal_loss = adversarial_loss(d_unreal, real)

            con_loss = self.content_loss(g_out, src)
            g_loss = d_unreal_loss + self.hparams.con_lambda * con_loss
            self.log("g_loss", g_loss)
            return g_loss

        if optimizer_idx == 1:
            # train discriminator
            self.set_requires_grad(self.discriminator, True)
            d_real = self.discriminator(tgt)
            d_real_loss = adversarial_loss(d_real, real)

            g_out = self.generator(src)
            d_unreal = self.discriminator(g_out.detach())
            d_unreal_loss = adversarial_loss(d_unreal, unreal)

            d_edge = self.discriminator(edge)
            d_edge_loss = adversarial_loss(d_edge, unreal)

            d_loss = d_real_loss + d_unreal_loss + d_edge_loss
            self.log("d_loss", d_loss)
            return d_loss

    def validation_step(self, batch, batch_idx):
        # log sampled images
        g_out = self.generator(batch)
        grid = torchvision.utils.make_grid(g_out)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

    def configure_optimizers(self):
        opt_g = Adam(self.generator.parameters(), lr=self.hparams.lr_g, betas=(self.hparams.b1, self.hparams.b2))
        opt_d = Adam(self.discriminator.parameters(), lr=self.hparams.lr_d, betas=(self.hparams.b1, self.hparams.b2))

        g_scheduler = MultiStepLR(optimizer=opt_g, milestones=[self.trainer.max_epochs // 2, self.trainer.max_epochs //
                                                               4 * 3], gamma=0.1)
        d_scheduler = MultiStepLR(optimizer=opt_d, milestones=[self.trainer.max_epochs // 2, self.trainer.max_epochs //
                                                               4 * 3], gamma=0.1)

        return [opt_g, opt_d], [g_scheduler, d_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Cartoon GAN")
        parser.add_argument('--con_lambda', type=float, default=0.5)
        parser.add_argument('--lr_g', type=float, default=0.0002, help='learning rate generator')
        parser.add_argument('--lr_d', type=float, default=0.0002, help='learning rate discriminator')
        return parent_parser
