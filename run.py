from argparse import ArgumentParser

import torchvision.transforms
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer

from cg_module import CGModule
from data.data_module import DataModule
from pretrain_generator import PretrainGeneratorModule


def run_main(args):
    dict_args = vars(args)
    pretrain_g = PretrainGeneratorModule(**dict_args)  # Note: optionally use this module to pretrain generator
    cg_module = CGModule(**dict_args)

    data_module = DataModule(args.batch_size, args.n_workers, args.src_data, args.tgt_data)
    trainer = Trainer.from_argparse_args(args, gpus=1)   # , max_epochs=dict_args["pre_train_epoch"]
    trainer.fit(cg_module, data_module)


def main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument('--b1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--b2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--in_nc', type=int, default=3)
    parser.add_argument('--out_nc', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--n_workers', type=int, default=32, help='workers for data loader')
    parser.add_argument('--pre_train_epoch', type=int, default=10)
    parser.add_argument('--src_data', required=False, default='src_data_path',
                        help='sec data path')
    parser.add_argument('--tgt_data', required=False, default='tgt_data_path',
                        help='tgt data path')

    parser = CGModule.add_model_specific_args(parser)
    run_main(parser.parse_args())


if __name__ == '__main__':
    main()
