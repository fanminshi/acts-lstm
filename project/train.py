from argparse import ArgumentParser

from cycler import L
import torch
import lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchtext.vocab import GloVe, vocab, build_vocab_from_iterator
from nltk.tokenize import word_tokenize
from torch.optim.lr_scheduler import ExponentialLR
from lightning.pytorch.callbacks import LearningRateMonitor


# Training parameters
EMB_DIM = 300
LSTM_HIDDEN_DIM = 2048
LATENT_DIM = 512
N_CLASSES = 3
_UNK = "<unk>"


class BaselineEncoder(nn.Module):
    def __init__(self, pretrained_embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=True
        )

    def forward(self, input_ids):
        return torch.mean(self.embedding(input_ids), dim=1)


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        pretrained_embeddings,
        input_dim,
        latent_dim,
        bidirectional=False,
        dropout=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=True
        )
        self.hidden_dim = latent_dim
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=latent_dim,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, input_ids, text_lengths):
        embeds = self.embedding(input_ids)
        batch_size = input_ids.size(0)
        packed_embedded = pack_padded_sequence(embeds, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.rnn(packed_embedded)
        return torch.squeeze(torch.permute(hn, (1, 0, 2)))


class NLI(pl.LightningModule):
    def __init__(self, enc, mlp_input_dim, latent_dim, n_classes):
        super().__init__()
        self.encoder = enc
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim * 4, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, n_classes),
        )

    def forward(self, premise_ids, premise_len, hypothesis_ids, hypothesis_len):
        # in lightning, forward defines the prediction/inference actions
        p_embed = self.encoder(premise_ids, premise_len)
        h_embed = self.encoder(hypothesis_ids, hypothesis_len)
        embed = torch.cat(
            (p_embed, h_embed, torch.abs(p_embed - h_embed), p_embed * h_embed), dim=1
        )
        return self.mlp(embed)

    def training_step(self, batch, batch_idx):
        premise, premise_len, hypothesis, hypothesis_len, y = batch
        y_logits = self(premise, premise_len, hypothesis, hypothesis_len)
        loss = F.cross_entropy(y_logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        premise, premise_len, hypothesis, hypothesis_len, y = batch
        y_logits = self(premise, premise_len, hypothesis, hypothesis_len)
        loss = F.cross_entropy(y_logits, y)
        self.log(f"eval_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=512, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    train_ds, val_ds, test_ds = load_dataset(
        "stanfordnlp/snli", split=["train", "validation", "test"]
    )

    def lower(words):
        return [w.lower() for w in word_tokenize(words)]

    def preprocess(example):
        tokens = lower(example["premise"])
        example["premise"] = tokens
        tokens = lower(example["hypothesis"])
        example["hypothesis"] = tokens
        return example

    train_ds = train_ds.map(preprocess)
    val_ds = val_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)

    unique_tokens = set()

    def update_unique_tokens(ds):
        for item in ds:
            unique_tokens.update((item["premise"]))
            unique_tokens.update((item["hypothesis"]))

    update_unique_tokens(train_ds)
    update_unique_tokens(val_ds)
    update_unique_tokens(test_ds)
    print("unique token len", len(unique_tokens))

    glove_vectors = GloVe(name="840B", dim=EMB_DIM)
    tokens_list = list(unique_tokens)
    tokens_list.sort()
    glove_vocab = build_vocab_from_iterator(
        [tokens_list], specials=[_UNK], special_first=True
    )
    glove_vocab.set_default_index(0)
    tokens_list.sort()
    pretrained_embeddings = glove_vectors.get_vecs_by_tokens([_UNK] + tokens_list)

    def collate_fn(batch):
        premise_tensors = []
        hypothesis_tensors = []
        label_tensors = []
        premise_text_lengths = []
        hypothesis_text_lengths = []
        for b in batch:
            p = b["premise"]
            h = b["hypothesis"]
            l = b["label"]
            if l == -1:
                continue
            p_indices = glove_vocab.lookup_indices(p)
            premise_text_lengths.append(len(p))
            premise_tensors.append(torch.tensor(p_indices))
            h_indices = glove_vocab.lookup_indices(h)
            hypothesis_text_lengths.append(len(h))
            hypothesis_tensors.append(torch.tensor(h_indices))
            label_tensors.append(l)

        premise_text = torch.tensor(premise_text_lengths)
        hypothesis_text = torch.tensor(hypothesis_text_lengths)
        # print("premise_text_lengths",  premise_text.size(), "hypothesis_text_lengths", hypothesis_text.size())
        premise = pad_sequence(premise_tensors, batch_first=True)
        hypothesis = pad_sequence(hypothesis_tensors, batch_first=True)
        return premise, premise_text, hypothesis, hypothesis_text, torch.tensor(label_tensors)

    train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=24)
    val_dl = DataLoader(val_ds, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=24)
    # ------------
    # model
    # ------------

    # BaselineEncoder(pretrained_embeddings=pretrained_embeddings)
    encoder = LSTMEncoder(
        pretrained_embeddings=pretrained_embeddings,
        input_dim=EMB_DIM,
        latent_dim=LSTM_HIDDEN_DIM,
    )
    model = NLI(
        enc=encoder,
        mlp_input_dim=LSTM_HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        n_classes=N_CLASSES,
    )

    # # ------------
    # # training
    # # ------------
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        max_epochs=10,
        val_check_interval=0.25,
        callbacks=[lr_monitor],
    )
    trainer.fit(model, train_dl, val_dl)

    # # ------------
    # # testing
    # # ------------
    # trainer.test(model, dataloaders=test_dl)


if __name__ == "__main__":
    cli_main()
