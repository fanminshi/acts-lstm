from argparse import ArgumentParser

from cycler import L
import torch
import lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchtext.vocab import GloVe, vocab, build_vocab_from_iterator
from nltk.tokenize import word_tokenize
from torch.optim.lr_scheduler import ExponentialLR
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics import Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
from SentEval import senteval
import logging

import inspect

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec



# Training parameters
EMB_DIM = 300
LSTM_HIDDEN_DIM = 2048
LATENT_DIM = 512
N_CLASSES = 3
_UNK = "<unk>"
PATH_TO_DATA='./SentEval/data/'

glove_vectors = GloVe(name="840B", dim=EMB_DIM)
def prepare(params, samples):

    unique_tokens = set()
    for s in samples:
        unique_tokens.update(s)
    tokens_list = list(unique_tokens)
    print("prepare uniq tokens", len(tokens_list))
    tokens_list.sort()
    glove_vocab = build_vocab_from_iterator(
        [tokens_list], specials=[_UNK], special_first=True
    )
    glove_vocab.set_default_index(0)
    pretrained_embeddings = glove_vectors.get_vecs_by_tokens([_UNK] + tokens_list)
    params['glove_vocab'] = glove_vocab
    params.infersent.encoder.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=True).cuda()


def batcher(params, batch):
    sentences = []
    text_lengths = []
    for s in batch:
        if len(s) == 0:
            s=['<unk>']
        s_indices = params['glove_vocab'].lookup_indices(s)
        text_lengths.append(len(s))
        sentences.append(torch.tensor(s_indices, dtype=torch.int32))

    text = torch.tensor(text_lengths).cuda()
    sen_pad = pad_sequence(sentences, batch_first=True)
    sen_pad = sen_pad.cuda()
    embeddings = params.infersent.encoder(sen_pad, text)
    return embeddings.cpu().detach().numpy()


class BaselineEncoder(nn.Module):
    def __init__(self, pretrained_embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=True
        )

    def forward(self, input_ids, text_lengths):
        # [B, EMB_DIM]
        s = torch.sum(self.embedding(input_ids), dim=1)
        # [B] -> [B, 1] for broadcasting
        text_lengths = text_lengths.view(-1, 1)
        return s / text_lengths


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        pretrained_embeddings,
        input_dim,
        latent_dim,
        bidirectional=False,
        max_pool=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=True
        )
        self.hidden_dim = latent_dim
        self.max_pool = max_pool
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=latent_dim,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, input_ids, text_lengths):
        embeds = self.embedding(input_ids)
        packed_embedded = pack_padded_sequence(
            embeds, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, (hn, cn) = self.rnn(packed_embedded)
        if not self.max_pool:
            # S, B, D -> B, S, D
            hn = torch.permute(hn, (1, 0, 2))
            return hn.reshape(hn.size(0), hn.size(1) * hn.size(2))
        # Batch_Size, Seq_Len, Emed_DIM
        emb_unpacked, lens_unpacked = pad_packed_sequence(output, batch_first=True)
        output = [x[:l] for x, l in zip(emb_unpacked, lens_unpacked)]
        emb = [torch.max(x, 0)[0] for x in output]
        emb = torch.stack(emb, 0)
        return emb


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
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)

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
        acc = self.accuracy(y_logits, y)
        self.log("val_accuracy", acc, on_epoch=True)
        self.log(f"val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        premise, premise_len, hypothesis, hypothesis_len, y = batch
        y_logits = self(premise, premise_len, hypothesis, hypothesis_len)
        loss = F.cross_entropy(y_logits, y)
        acc = self.accuracy(y_logits, y)
        self.log("test_accuracy", acc)
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]

def build_model(args, pretrained_embeddings, checkpt):
    encoder = None
    if args.encoder == "avg":
        encoder = BaselineEncoder(pretrained_embeddings=pretrained_embeddings)
        mlp_input_dim = EMB_DIM
    if args.encoder == "lstm":
        encoder = LSTMEncoder(
            pretrained_embeddings=pretrained_embeddings,
            input_dim=EMB_DIM,
            latent_dim=LSTM_HIDDEN_DIM,
        )
        mlp_input_dim = LSTM_HIDDEN_DIM
    if args.encoder == "bi-lstm":
        encoder = LSTMEncoder(
            pretrained_embeddings=pretrained_embeddings,
            input_dim=EMB_DIM,
            latent_dim=LSTM_HIDDEN_DIM,
            bidirectional=True,
        )
        mlp_input_dim = LSTM_HIDDEN_DIM * 2
    if args.encoder == "bi-lstm-max-pool":
        encoder = LSTMEncoder(
            pretrained_embeddings=pretrained_embeddings,
            input_dim=EMB_DIM,
            latent_dim=LSTM_HIDDEN_DIM,
            bidirectional=True,
            max_pool=True,
        )
        mlp_input_dim = LSTM_HIDDEN_DIM * 2
    if checkpt:
        return NLI.load_from_checkpoint(checkpt, enc=encoder, mlp_input_dim=mlp_input_dim, latent_dim=LATENT_DIM, n_classes=N_CLASSES)
    return NLI(
        enc=encoder,
        mlp_input_dim=mlp_input_dim,
        latent_dim=LATENT_DIM,
        n_classes=N_CLASSES,
    )

def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--senteval", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--encoder",
        default="avg",
        choices=["avg", "lstm", "bi-lstm", "bi-lstm-max-pool"],
        type=str,
    )
    parser.add_argument("--save_dir", default="./model", type=str)
    parser.add_argument("--checkpt", type=str)
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
    tokens_list = list(unique_tokens)
    tokens_list.sort()
    glove_vocab = build_vocab_from_iterator(
        [tokens_list], specials=[_UNK], special_first=True
    )
    glove_vocab.set_default_index(0)
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
        premise = pad_sequence(premise_tensors, batch_first=True)
        hypothesis = pad_sequence(hypothesis_tensors, batch_first=True)
        return (
            premise,
            premise_text,
            hypothesis,
            hypothesis_text,
            torch.tensor(label_tensors),
        )

    train_dl = DataLoader(
        train_ds, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=24
    )
    val_dl = DataLoader(
        val_ds, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=24
    )
    test_dl = DataLoader(
        test_ds, collate_fn=collate_fn, batch_size=args.batch_size
    )
    # ------------
    # model
    # ------------

    model = build_model(args, pretrained_embeddings, "")
    # # ------------
    # # SentEval
    # # ------------
    if args.senteval:
        logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 2}
        params['infersent'] = build_model(args, pretrained_embeddings, args.checkpt).cuda()
        se = senteval.SE(params, batcher, prepare)
        transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'SICKEntailment']
        results = se.eval(transfer_tasks)
        print(results)
        return

    # # ------------
    # # training
    # # ------------
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # early stop when eval loss hasn't decreased 3 times in a row.
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    checkpt_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=f"{args.encoder}-" + "{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    logger = TensorBoardLogger(
        args.save_dir, name=f"{args.encoder}"
    )
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir=args.save_dir,
        accelerator="gpu",
        devices="auto",
        max_epochs=20,
        callbacks=[lr_monitor, early_stop_callback, checkpt_callback],
    )
    trainer.fit(model, train_dl, val_dl)

    # # ------------
    # # testing
    # # ------------
    trainer.test(model, dataloaders=test_dl)



if __name__ == "__main__":
    cli_main()
