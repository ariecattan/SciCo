import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import socket
import yaml
from torch.utils import data
from tqdm import tqdm
import os
from transformers import AutoTokenizer
import logging

from utils.utils import *
from models.datasets import CrossEncoderDataset
from models.muticlass import MulticlassModel


def init_logs():
    root_logger = logging.getLogger()
    logger = root_logger.getChild(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=os.path.join(
        config['log'], '{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return logger



def get_sep_tokens(bert_model):
    if 'longformer' in bert_model or 'cdlm' in bert_model.lower():
        return '</s>'
    return '[SEP]'


def get_train_dev_loader(config):
    logger.info('Loading data')
    sep_token = get_sep_tokens(config['model']['bert_model'])
    cdlm = 'cdlm' in config['model']['bert_model'].lower()

    train = CrossEncoderDataset(config["data"]["training_set"], full_doc=config['full_doc'], multiclass=model_name,
                                cdlm=cdlm, sep_token=sep_token)
    train_loader = data.DataLoader(train,
                                   batch_size=config["model"]["batch_size"],
                                   shuffle=True,
                                   collate_fn=model.tokenize_batch,
                                   num_workers=16,
                                   pin_memory=True)
    logger.info(f'training size: {len(train)}')

    dev = CrossEncoderDataset(config["data"]["dev_set"], full_doc=config['full_doc'], multiclass=model_name, cdlm=cdlm,
                              sep_token=sep_token)
    dev_loader = data.DataLoader(dev,
                                 batch_size=config["model"]["batch_size"],
                                 shuffle=False,
                                 collate_fn=model.tokenize_batch,
                                 num_workers=16,
                                 pin_memory=True)
    logger.info(f'Validation size: {len(dev)}')

    return train_loader, dev_loader



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/multiclass.yaml')
    parser.add_argument('--multiclass', type=str, default='multiclass')
    parser.add_argument('--full_doc', type=str, default='0')
    parser.add_argument('--bert_model', type=str, default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_name = args.multiclass

    config['full_doc'] = False if args.full_doc == '0' else True
    config['model']['bert_model'] = args.bert_model if args.bert_model != '' else config['model']['bert_model']

    if model_name not in {'coref', 'hypernym', 'multiclass'}:
        raise ValueError(f"The multiclass value needs to be in (coref, hypernym, multiclass), got {model_name}.")

    logger = init_logs()
    logger.info("pid: {}".format(os.getpid()))
    logger.info('Server name: {}'.format(socket.gethostname()))
    fix_seed(config["random_seed"])
    create_folder(config['model_path'])

    logger.info(f"Training {model_name}")
    logger.info('loading models')
    model = MulticlassModel.get_model(model_name, config)
    train_loader, dev_loader = get_train_dev_loader(config)
    pl_logger = CSVLogger(save_dir=config['model_path'], name=model_name)
    pl_logger.log_hyperparams(config)
    checkpoint_callback = ModelCheckpoint(save_top_k=-1)
    trainer = pl.Trainer(gpus=config['gpu_num'],
                         default_root_dir=config['model_path'],
                         accelerator='ddp',
                         plugins=DDPPlugin(find_unused_parameters=False),
                         max_epochs=config['model']['epochs'],
                         callbacks=[checkpoint_callback],
                         logger=pl_logger,
                         gradient_clip_val=config['model']['gradient_clip'],
                         accumulate_grad_batches=config['model']['gradient_accumulation'],
                         val_check_interval=1.0)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=dev_loader)