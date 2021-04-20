import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pytorch_lightning as pl
import torchmetrics


class EntailmentModel(pl.LightningModule):
    '''
    0 not related (contradiction)
    1 hypernym (neutral)
    2 hyponym (entailment)
    '''

    def __init__(self, config):
        super(EntailmentModel, self).__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
        self.model = AutoModelForSequenceClassification.from_pretrained(config['nli_model'])


        self.acc = pl.metrics.Accuracy(top_k=1)
        self.f1 = pl.metrics.F1(num_classes=3, average='none')
        self.recall = pl.metrics.Recall(num_classes=3, average='none')
        self.val_precision = pl.metrics.Precision(num_classes=3, average='none')


    def forward(self, input_ids, attention_mask):
        scores = self.model(input_ids, attention_mask).logits
        return scores

        # scores = torch.softmax(scores, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask = x
        y_hat = self(input_ids, attention_mask)
        loss = self.criterion(y_hat, y)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask = x
        y_hat = self(input_ids, attention_mask)
        loss = self.criterion(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=1)
        self.compute_metrics(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        self.log('accuracy', self.acc.compute())
        prec_neg, prec_neutral, prec_entailment = self.val_precision.compute()
        _, _, recall_entailment = self.recall.compute()
        _, _, f1_entailment = self.f1.compute()
        self.log('precision', prec_entailment)
        self.log('recall', recall_entailment)
        self.log('f1', f1_entailment)


    def compute_metrics(self, y_hat, y):
        self.acc(y_hat, y)
        self.f1(y_hat, y)
        self.recall(y_hat, y)
        self.val_precision(y_hat, y)




    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])


    def set_loss(self, counter):
        weights = 1 / torch.tensor([counter[0], counter[1], counter[2]])
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))

    def tokenize_batch(self, batch):
        premise, hypothesis, labels = zip(*batch)
        tokens = self.tokenizer(premise, hypothesis, padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        labels = torch.tensor(labels)

        return (input_ids, attention_mask), labels
