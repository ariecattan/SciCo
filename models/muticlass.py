import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import *
import numpy as np
import pytorch_lightning as pl
import torchmetrics

from typing import Any, List, Optional


def get_global_attention(input_ids, start_token, end_token):
    global_attention_mask = torch.zeros(input_ids.shape)
    global_attention_mask[:, 0] = 1  # global attention to the CLS token
    start = torch.nonzero(input_ids == start_token)
    end = torch.nonzero(input_ids == end_token)
    globs = torch.cat((start, end))
    value = torch.ones(globs.shape[0])
    global_attention_mask.index_put_(tuple(globs.t()), value)
    return global_attention_mask




class MulticlassModel:
    def __init__(self):
        super(MulticlassModel, self).__init__()


    @classmethod
    def get_model(cls, name, config):
        if name == 'multiclass':
            return MulticlassCrossEncoder(config, num_classes=4)
        elif name == 'coref':
            return BinaryCorefCrossEncoder(config)
        elif name == 'hypernym':
            return HypernymCrossEncoder(config)




class MulticlassCrossEncoder(pl.LightningModule):
    '''
    multiclass classification with labels:
    0 not related
    1 coref
    2. hypernym
    3. neutral (hyponym)
    '''

    def __init__(self, config, num_classes=4):
        super(MulticlassCrossEncoder, self).__init__()
        self.cdlm = 'cdlm' in config["model"]["bert_model"].lower()
        self.long = True if 'longformer' in config["model"]["bert_model"] or self.cdlm else False
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')
        self.sep = self.tokenizer.convert_tokens_to_ids('</s>')
        self.doc_start = self.tokenizer.convert_tokens_to_ids('<doc-s>') if self.cdlm else None
        self.doc_end = self.tokenizer.convert_tokens_to_ids('</doc-s>') if self.cdlm else None

        self.model = AutoModel.from_pretrained(config["model"]["bert_model"], add_pooling_layer=False)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()


        self.acc = pl.metrics.Accuracy(top_k=1)
        self.f1 = pl.metrics.F1(num_classes=num_classes, average='none')
        self.recall = pl.metrics.Recall(num_classes=num_classes, average='none')
        self.val_precision = pl.metrics.Precision(num_classes=num_classes, average='none')



    def forward(self, input_ids, attention_mask, global_attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask,
                            global_attention_mask=global_attention_mask)
        cls_vector = output.last_hidden_state[:, 0, :]
        scores = self.linear(cls_vector)
        return scores



    def training_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        return loss



    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=1)
        self.compute_metrics(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)

        return loss


    def validation_epoch_end(self, outputs):
        self.log_metrics()


    def test_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=1)

        return {
            'loss': loss,
            'preds': y_hat,
            'label': y
        }


    def test_step_end(self, outputs):
        y_hat, y = outputs['preds'], outputs['label']
        self.compute_metrics(y_hat, y)
        return outputs


    def test_epoch_end(self, outputs):
        self.log_metrics()
        self.results = outputs




    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        y_hat = torch.softmax(y_hat, dim=1)
        return y_hat




    def compute_metrics(self, y_hat, y):
        self.acc(y_hat, y)
        self.f1(y_hat, y)
        self.recall(y_hat, y)
        self.val_precision(y_hat, y)




    def log_metrics(self):
        self.log('acc', self.acc.compute())
        f1_negative, f1_coref, f1_hypernym, f1_hyponym = self.f1.compute()
        recall_negative, recall_coref, recall_hypernym, recall_hyponym = self.recall.compute()
        precision_negative, precision_coref, precision_hypernym, precision_hyponym = self.val_precision.compute()
        self.log('f1_coref', f1_coref)
        self.log('recall_coref', recall_coref)
        self.log('precision_coref', precision_coref)
        self.log('f1_hypernym', f1_hypernym)
        self.log('recall_hypernym', recall_hypernym)
        self.log('precision_hypernym', precision_hypernym)
        self.log('f1_hyponym', f1_hyponym)
        self.log('recall_hyponym', recall_hyponym)
        self.log('precision_hyponym', precision_hyponym)





    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])

    def get_global_attention(self, input_ids):
        global_attention_mask = torch.zeros(input_ids.shape)
        global_attention_mask[:, 0] = 1  # global attention to the CLS token
        start = torch.nonzero(input_ids == self.start)
        end = torch.nonzero(input_ids == self.end)
        if self.cdlm:
            doc_start = torch.nonzero(input_ids == self.doc_start)
            doc_end = torch.nonzero(input_ids == self.doc_end)
            globs = torch.cat((start, end, doc_start, doc_end))
        else:
            globs = torch.cat((start, end))

        value = torch.ones(globs.shape[0])
        global_attention_mask.index_put_(tuple(globs.t()), value)
        return global_attention_mask


    def tokenize_batch(self, batch):
        inputs, labels = zip(*batch)
        tokens = self.tokenizer(list(inputs), padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        global_attention_mask = self.get_global_attention(input_ids)
        labels = torch.stack(labels)

        return (input_ids, attention_mask, global_attention_mask), labels



class BinaryCorefCrossEncoder(pl.LightningModule):
    '''
    multiclass classification with labels:
    0 not related, hypernyn, or hyponym
    1 coref
    '''

    def __init__(self, config):
        super(BinaryCorefCrossEncoder, self).__init__()
        self.long = True if 'longformer' in config["model"]["bert_model"] else False
        self.config = config


        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')

        self.model = AutoModel.from_pretrained(config["model"]["bert_model"], add_pooling_layer=False)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()


        self.acc = pl.metrics.Accuracy()
        self.f1 = pl.metrics.F1(num_classes=1)
        self.recall = pl.metrics.Recall(num_classes=1)
        self.val_precision = pl.metrics.Precision(num_classes=1)



    def forward(self, input_ids, attention_mask, global_attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask,
                            global_attention_mask=global_attention_mask)
        cls_vector = output.last_hidden_state[:, 0, :]
        scores = self.linear(cls_vector)
        return scores.squeeze(1)



    def training_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        return loss



    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        y_hat = torch.sigmoid(y_hat)
        self.compute_metrics(y_hat, y.to(torch.int))
        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)

        return loss


    def validation_epoch_end(self, outputs):
        self.log_metrics()


    def test_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)

        return {
            'loss': loss,
            'preds': y_hat,
            'label': y
        }


    def test_step_end(self, outputs):
        y_hat, y = outputs['preds'], outputs['label']
        self.compute_metrics(y_hat, y)
        return outputs


    def test_epoch_end(self, outputs):
        self.log_metrics()
        self.results = outputs



    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        y_hat = torch.sigmoid(y_hat)
        return y_hat




    def compute_metrics(self, y_hat, y):
        self.acc(y_hat, y)
        self.f1(y_hat, y)
        self.recall(y_hat, y)
        self.val_precision(y_hat, y)




    def log_metrics(self):
        self.log('acc', self.acc.compute())
        self.log('f1', self.f1.compute())
        self.log('recall', self.recall.compute())
        self.log('precision', self.val_precision.compute())





    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])



    def tokenize_batch(self, batch):
        inputs, labels = zip(*batch)
        tokens = self.tokenizer(list(inputs), padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        global_attention_mask = get_global_attention(input_ids, self.start, self.end)
        labels = torch.stack(labels)

        return (input_ids, attention_mask, global_attention_mask), labels



class HypernymCrossEncoder(pl.LightningModule):
    '''
        multiclass classification with labels:
        0 not related or coref
        1. hypernym
        2. hyponym
        '''

    def __init__(self, config, num_classes=3):
        super(HypernymCrossEncoder, self).__init__()
        self.long = True if 'longformer' in config["model"]["bert_model"] else False
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')

        self.model = AutoModel.from_pretrained(config["model"]["bert_model"], add_pooling_layer=False)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.acc = pl.metrics.Accuracy(top_k=1)
        self.f1 = pl.metrics.F1(num_classes=num_classes, average='none')
        self.recall = pl.metrics.Recall(num_classes=num_classes, average='none')
        self.val_precision = pl.metrics.Precision(num_classes=num_classes, average='none')


    def forward(self, input_ids, attention_mask, global_attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask,
                            global_attention_mask=global_attention_mask)
        cls_vector = output.last_hidden_state[:, 0, :]
        scores = self.linear(cls_vector)
        return scores


    def training_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=1)
        self.compute_metrics(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)

        return loss


    def validation_epoch_end(self, outputs):
        self.log_metrics()


    def test_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=1)

        return {
            'loss': loss,
            'preds': y_hat,
            'label': y
        }

    def test_step_end(self, outputs):
        y_hat, y = outputs['preds'], outputs['label']
        self.compute_metrics(y_hat, y)
        return outputs

    def test_epoch_end(self, outputs):
        self.log_metrics()
        self.results = outputs


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        y_hat = torch.softmax(y_hat, dim=1)
        return y_hat


    def compute_metrics(self, y_hat, y):
        self.acc(y_hat, y)
        self.f1(y_hat, y)
        self.recall(y_hat, y)
        self.val_precision(y_hat, y)


    def log_metrics(self):
        self.log('acc', self.acc.compute())
        f1_negative, f1_hypernym, f1_hyponym = self.f1.compute()
        recall_negative, recall_hypernym, recall_hyponym = self.recall.compute()
        precision_negative, precision_hypernym, precision_hyponym = self.val_precision.compute()
        self.log('f1_hypernym', f1_hypernym)
        self.log('recall_hypernym', recall_hypernym)
        self.log('precision_hypernym', precision_hypernym)
        self.log('f1_hyponym', f1_hyponym)
        self.log('recall_hyponym', recall_hyponym)
        self.log('precision_hyponym', precision_hyponym)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])


    def tokenize_batch(self, batch):
        inputs, labels = zip(*batch)
        tokens = self.tokenizer(list(inputs), padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        global_attention_mask = get_global_attention(input_ids, self.start, self.end)
        labels = torch.stack(labels)

        return (input_ids, attention_mask, global_attention_mask), labels



class MulticlassBiEncoder(pl.LightningModule):
    def __init__(self, config, num_classes=4):
        super(MulticlassBiEncoder, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.long = 'longformer' in config['model']['bert_model']

        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')

        self.model = AutoModel.from_pretrained(config["model"]["bert_model"], add_pooling_layer=False)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size * 2, num_classes)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.acc = pl.metrics.Accuracy(top_k=1)
        self.f1 = pl.metrics.F1(num_classes=num_classes, average='none')
        self.recall = pl.metrics.Recall(num_classes=num_classes, average='none')
        self.val_precision = pl.metrics.Precision(num_classes=num_classes, average='none')


    def get_cls_token(self, mention):
        input_ids, attention_mask, global_attention_mask = mention
        if self.long:
            output = self.model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        else:
            output = self.model(input_ids, attention_mask=attention_mask)

        return output.last_hidden_state[:, 0, :]


    def forward(self, first, second):
        cls_1 = self.get_cls_token(first)
        cls_2 = self.get_cls_token(second)

        input_vec = torch.cat((cls_1, cls_2), dim=1)
        scores = self.linear(input_vec)
        return scores


    def training_step(self, batch, batch_idx):
        m1, m2, y = batch
        y_hat = self(m1, m2)
        loss = self.criterion(y_hat, y)
        return loss


    def validation_step(self, batch, batch_idx):
        m1, m2, y = batch
        y_hat = self(m1, m2)
        loss = self.criterion(y_hat, y)
        self.compute_metrics(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)

        return loss


    def validation_epoch_end(self, outputs):
        self.log_metrics()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        m1, m2, y = batch
        y_hat = self(m1, m2)
        y_hat = torch.softmax(y_hat, dim=1)
        return y_hat



    def compute_metrics(self, y_hat, y):
        self.acc(y_hat, y)
        self.f1(y_hat, y)
        self.recall(y_hat, y)
        self.val_precision(y_hat, y)


    def log_metrics(self):
        self.log('acc', self.acc.compute())
        f1_negative, f1_coref, f1_hypernym, f1_hyponym = self.f1.compute()
        recall_negative, recall_coref, recall_hypernym, recall_hyponym = self.recall.compute()
        precision_negative, precision_coref, precision_hypernym, precision_hyponym = self.val_precision.compute()
        self.log('f1_coref', f1_coref)
        self.log('recall_coref', recall_coref)
        self.log('precision_coref', precision_coref)
        self.log('f1_hypernym', f1_hypernym)
        self.log('recall_hypernym', recall_hypernym)
        self.log('precision_hypernym', precision_hypernym)
        self.log('f1_hyponym', f1_hyponym)
        self.log('recall_hyponym', recall_hyponym)
        self.log('precision_hyponym', precision_hyponym)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])


    def tokenize_mention(self, mentions):
        tokens = self.tokenizer(list(mentions), padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])

        if self.long:
            global_attention_mask = get_global_attention(input_ids, self.start, self.end)
        else:
            global_attention_mask = torch.tensor([])

        return input_ids, attention_mask, global_attention_mask


    def tokenize_batch(self, batch):
        first, second, labels = zip(*batch)
        m1 = self.tokenize_mention(first)
        m2 = self.tokenize_mention(second)
        labels = torch.stack(labels)

        return m1, m2, labels

