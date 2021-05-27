from argparse import ArgumentParser

from pytorch_lightning import Trainer

from cg_module import CGModule
from pretrain_generator import PretrainGeneratorModule


def main(args):
    dict_args = vars(args)
    pretrain_g = PretrainGeneratorModule(**dict_args)
    cg_module = CGModule(**dict_args)

    trainer = Trainer.from_argparse_args(args, max_epochs=dict_args["pre_train_epoch"])
    trainer.fit(pretrain_g)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument('--b1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--b2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--in_nc', type=int, default=3)
    parser.add_argument('--out_nc', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--n_workers', type=int, default=16, help='workers for data loader')
    parser.add_argument('--pre_train_epoch', type=int, default=10)
    parser.add_argument('--src_data', required=False, default='/home/winfried_loetzsch/data/ffhq_1000',
                        help='sec data path')
    parser.add_argument('--tgt_data', required=False, default='/home/winfried_loetzsch/data/anime',
                        help='tgt data path')

    parser = CGModule.add_model_specific_args(parser)
