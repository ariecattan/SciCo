import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import *
import numpy as np
import pytorch_lightning as pl
import torchmetrics

from typing import Any, List, Optional


class MulticlassCrossEncoder:
    def __init__(self):
        super(MulticlassCrossEncoder, self).__init__()


    @classmethod
    def get_model(cls, name, config):
        if name == 'multiclass':
            return CorefEntailmentLightning(config, num_classes=4)
        elif name == 'coref':
            return BinaryCorefEntailmentLightning(config)
        elif name == 'hypernym':
            return HypernymModel(config)



class HFMulticlassLightning(pl.LightningModule):
    def __init__(self, config, num_classes):
        super(HFMulticlassLightning, self).__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_model'])
        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')

        self.bert_config = AutoConfig.from_pretrained(config['model']['bert_model'])
        self.bert_config.num_labels = num_classes
        self.model = AutoModelForSequenceClassification(self.bert_config)






class CorefEntailmentLightning(pl.LightningModule):
    '''
    multiclass classification with labels:
    0 not related
    1 coref
    2. hypernym
    3. neutral (hyponym)
    '''

    def __init__(self, config, num_classes=4):
        super(CorefEntailmentLightning, self).__init__()
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
        # self.linear = nn.Sequential(
        #     nn.Linear(self.model.config.hidden_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_classes)
        # )
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
        y_hat = torch.softmax(y_hat, dim=1)
        loss = self.criterion(y_hat, y)
        self.compute_metrics(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)

        return loss


    def validation_epoch_end(self, outputs):
        self.log_metrics()


    def test_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        y_hat = torch.softmax(y_hat, dim=1)
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

        y_hat = torch.cat([x['preds'] for x in outputs])
        y = torch.cat([x['label'] for x in outputs])
        self.val_auroc(y_hat, y)
        auc = self.val_auroc.compute()
        neg, coref, hypernym, hyponym = auc
        # self.log('auc', auc.mean())
        self.log('auc_average', (neg + coref + hyponym + hypernym) / 4)
        self.log('auc_neg', neg)
        self.log('auc_coref', coref)
        self.log('auc_hyper', hypernym)
        self.log('auc_hypo', hyponym)



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
        # self.val_auroc(y_hat, y)




    def log_metrics(self):
        self.log('acc', self.acc.compute())
        f1_negative, f1_coref, f1_hypernym, f1_hyponym = self.f1.compute()
        recall_negative, recall_coref, recall_hypernym, recall_hyponym = self.recall.compute()
        precision_negative, precision_coref, precision_hypernym, precision_hyponym = self.val_precision.compute()
        # auc = self.val_auroc.compute()
        # auc_neg, auc_coref, auc_hypernym, auc_hyponym = self.auc.compute()
        self.log('f1_coref', f1_coref)
        self.log('recall_coref', recall_coref)
        self.log('precision_coref', precision_coref)
        self.log('f1_hypernym', f1_hypernym)
        self.log('recall_hypernym', recall_hypernym)
        self.log('precision_hypernym', precision_hypernym)
        self.log('f1_hyponym', f1_hyponym)
        self.log('recall_hyponym', recall_hyponym)
        self.log('precision_hyponym', precision_hyponym)
        # self.log('auc', auc)
        # self.log('auc_neg', auc_neg)
        # self.log('auc_coref', auc_coref)
        # self.log('auc_hypernym', auc_hypernym)
        # self.log('auc_hyponym', auc_hyponym)





    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])


    def get_global_attention(self, input_ids):
        global_attention_mask = torch.zeros(input_ids.shape)
        global_attention_mask[:, 0] = 1  # global attention to the CLS token
        start = torch.nonzero(input_ids == self.start)
        end = torch.nonzero(input_ids == self.end)
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



class BinaryCorefEntailmentLightning(pl.LightningModule):
    '''
    multiclass classification with labels:
    0 not related, hypernyn, or hyponym
    1 coref
    '''

    def __init__(self, config):
        super(BinaryCorefEntailmentLightning, self).__init__()
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
        # self.linear = nn.Sequential(
        #     nn.Linear(self.model.config.hidden_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )
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


    def get_global_attention(self, input_ids):
        global_attention_mask = torch.zeros(input_ids.shape)
        global_attention_mask[:, 0] = 1  # global attention to the CLS token
        start = torch.nonzero(input_ids == self.start)
        end = torch.nonzero(input_ids == self.end)
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



class HypernymModel(pl.LightningModule):
    '''
        multiclass classification with labels:
        0 not related or coref
        1. hypernym
        2. hyponym
        '''

    def __init__(self, config, num_classes=3):
        super(HypernymModel, self).__init__()
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
        # self.linear = nn.Sequential(
        #     nn.Linear(self.model.config.hidden_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_classes)
        # )
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
        y_hat = torch.softmax(y_hat, dim=1)
        loss = self.criterion(y_hat, y)
        self.compute_metrics(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)

        return loss


    def validation_epoch_end(self, outputs):
        self.log_metrics()


    def test_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        y_hat = torch.softmax(y_hat, dim=1)
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
        y_hat = torch.softmax(y_hat, dim=1)
        return y_hat


    def compute_metrics(self, y_hat, y):
        self.acc(y_hat, y)
        self.f1(y_hat, y)
        self.recall(y_hat, y)
        self.val_precision(y_hat, y)
        # self.val_auroc(y_hat, y)


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

    def get_global_attention(self, input_ids):
        global_attention_mask = torch.zeros(input_ids.shape)
        global_attention_mask[:, 0] = 1  # global attention to the CLS token
        start = torch.nonzero(input_ids == self.start)
        end = torch.nonzero(input_ids == self.end)
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