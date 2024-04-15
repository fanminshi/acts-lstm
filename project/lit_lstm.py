from argparse import ArgumentParser

import torch
import lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset


class LitLSTM(pl.LightningModule):
    def __init__(self, hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitLSTM.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    train_ds, val_ds, test_ds = load_dataset("stanfordnlp/snli",  split=['train', 'validation', 'test'])
    train_dl = DataLoader(train_ds, batch_size=4)
    val_dl = DataLoader(val_ds, batch_size=4)
    test_dl = DataLoader(test_ds, batch_size=4)

    # ------------
    # model
    # ------------
    # model = LitClassifier(args.hidden_dim, args.learning_rate)

    # # ------------
    # # training
    # # ------------
    # trainer = pl.Trainer.from_argparse_args(args)
    # trainer.fit(model, train_loader, val_loader)

    # # ------------
    # # testing
    # # ------------
    # trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
