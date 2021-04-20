import argparse
import pyhocon
from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from itertools import combinations
import jsonlines

from models.baselines import SpanEmbedder, SimplePairWiseClassifier
from utils.evaluator import Evaluation
from utils.model_utils import *
from utils.utils import *
from utils.corpus import Corpus

import logging
root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)


def train_pairwise_classifier(config, pairwise_model, span_repr, span_embeddings,
                                    first, second, labels, criterion, optimizer):
    accumulate_loss = 0
    start_end_embeddings, continuous_embeddings, width = span_embeddings
    batch_size = config['batch_size']
    # width = width.to(device)

    idx = shuffle(list(range(len(first))))
    for i in range(0, len(first), batch_size):
        optimizer.zero_grad()
        indices = idx[i:i+batch_size]
        batch_first, batch_second = first[indices], second[indices]
        batch_labels = labels[indices].to(torch.float)
        g1 = span_repr(start_end_embeddings[batch_first],
                                [continuous_embeddings[k] for k in batch_first], width[batch_first])
        g2 = span_repr(start_end_embeddings[batch_second],
                                [continuous_embeddings[k] for k in batch_second], width[batch_second])
        scores = pairwise_model(g1, g2)
        loss = criterion(scores.squeeze(1), batch_labels)
        accumulate_loss += loss.item()
        loss.backward()
        optimizer.step()

        # torch.cuda.empty_cache()

    return accumulate_loss






def get_pairwise_labels(labels, is_training):
    first, second = zip(*list(combinations(range(len(labels)), 2)))
    first = torch.tensor(first)
    second = torch.tensor(second)
    pairwise_labels = (labels[first] != 0) & (labels[second] != 0) & \
                      (labels[first] == labels[second])

    if is_training:
        positives = torch.nonzero(pairwise_labels == 1).squeeze()
        positive_ratio = len(positives) / len(first)
        negatives = torch.nonzero(pairwise_labels != 1).squeeze()
        rands = torch.rand(len(negatives))
        rands = (rands < positive_ratio * 20).to(torch.long)
        sampled_negatives = negatives[torch.nonzero(rands).squeeze()]
        new_first = torch.cat((first[positives], first[sampled_negatives]))
        new_second = torch.cat((second[positives], second[sampled_negatives]))
        new_labels = torch.cat((pairwise_labels[positives], pairwise_labels[sampled_negatives]))
        first, second, pairwise_labels = new_first, new_second, new_labels


    pairwise_labels = pairwise_labels.to(torch.long).to(device)

    if config['loss'] == 'hinge':
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(-1, device=device))
    else:
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(0, device=device))
    torch.cuda.empty_cache()


    return first, second, pairwise_labels



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_pairwise.json')
    args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file(args.config)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    logger.info('{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))
    logger.info("pid: {}".format(os.getpid()))


    fix_seed(config.random_seed)
    create_folder(config['model_path'])
    device = torch.device('cuda:{}'.format(config.gpu_num[0])) if torch.cuda.is_available() else 'cpu'
    logger.info('Using device {}'.format(device))


    # init train and dev set
    tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    with jsonlines.open(config['training_set'], 'r') as f:
        train = [line for line in f]
    train = Corpus(train, tokenizer)

    with jsonlines.open(config['dev_set'], 'r') as f:
        dev = [line for line in f]
    dev = Corpus(dev, tokenizer)


    ## Model initiation
    logger.info('Init checkpoints')
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    config['bert_hidden_size'] = bert_model.config.hidden_size
    span_repr = SpanEmbedder(config, device).to(device)
    pairwise_model = SimplePairWiseClassifier(config).to(device)

    if config['training_method'] == "fine-tune":
        span_repr.load_state_dict(torch.load(config['span_repr_path'], map_location=device))
        pairwise_model.load_state_dict(torch.load(config['pairwise_scorer_path'], map_location=device))


    ## Optimizer and loss function
    models = [span_repr, pairwise_model]
    optimizer = get_optimizer(config, models)
    criterion = get_loss_function(config)


    logger.info('Number of parameters of mention extractor: {}'.format(count_parameters(span_repr)))
    logger.info('Number of parameters of the pairwise classifier: {}'.format(count_parameters(pairwise_model)))

    logger.info('Number of topics: {}'.format(len(train.topics)))
    f1 = []
    for epoch in range(config['epochs']):
        logger.info('Epoch: {}'.format(epoch))

        pairwise_model.train()
        span_repr.train()

        accumulate_loss = 0
        # list_of_topics = shuffle(list(range(len(train.topics))))
        # list_of_topics = list(range(len(train.topics)))
        total_number_of_pairs = 0
        for topic_num, topic in enumerate(tqdm(train.topics)):
            doc_num, docs_embeddings, docs_length = pad_and_read_bert(topic['bert_tokens'], bert_model)
            continuous_embeddings, width, clusters = get_mention_embeddings(topic, docs_embeddings)


            #sanity check
            if min([len(x) for x in continuous_embeddings]) == 0:
                continue

            start_end = torch.stack([torch.cat((mention[0], mention[-1])) for mention in continuous_embeddings])
            width = torch.tensor(width, device=device)
            clusters = torch.tensor(clusters, device=device)
            span_embeddings = start_end, continuous_embeddings, width

            first, second = zip(*list(combinations(range(len(clusters)), 2)))
            first, second = torch.tensor(first), torch.tensor(second)
            pairwise_labels = clusters[first] == clusters[second]
            loss = train_pairwise_classifier(config, pairwise_model, span_repr, span_embeddings,
                                             first, second, pairwise_labels, criterion, optimizer)


            accumulate_loss += loss
            total_number_of_pairs += len(first)

        logger.info('Number of training pairs: {}'.format(total_number_of_pairs))
        logger.info('Accumulate loss: {}'.format(accumulate_loss))


        # logger.info('Evaluate on the dev set')
        # span_repr.eval()
        # pairwise_model.eval()
        # all_scores, all_labels = [], []
        #
        # for topic_num, topic in enumerate(tqdm(dev.topics)):
        #     topic = dev.topics[topic_num]
        #     doc_num, docs_embeddings, docs_length = pad_and_read_bert(topic['bert_tokens'], bert_model)
        #     continuous_embeddings, width, clusters = get_mention_embeddings(topic, docs_embeddings)
        #     start_end = torch.stack([torch.cat((mention[0], mention[-1])) for mention in continuous_embeddings])
        #     width = torch.tensor(width, device=device)
        #     clusters = torch.tensor(clusters, device=device)
        #
        #     with torch.no_grad():
        #         mention_embeddings = span_repr(start_end, continuous_embeddings, width)
        #
        #     pairwise_predictions, pairwise_labels = get_pairwise_scores(mention_embeddings, clusters, pairwise_model)
        #     # eval = Evaluation(pairwise_predictions, pairwise_labels)
        #     all_scores.extend(pairwise_predictions.squeeze(1))
        #     all_labels.extend(pairwise_labels.to(torch.int))
        #
        # all_labels = torch.stack(all_labels)
        # all_scores = torch.stack(all_scores)
        #
        #
        # strict_preds = (all_scores > 0).to(torch.int)
        # eval = Evaluation(strict_preds, all_labels.to(device))
        # logger.info('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        # logger.info('Number of positive pairs: {}/{}'.format(len(torch.nonzero(all_labels == 1)),
        #                                                      len(all_labels)))
        # logger.info('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
        #                                                                 eval.get_precision(), eval.get_f1()))
        # f1.append(eval.get_f1())
        torch.save(span_repr.state_dict(), os.path.join(config['model_path'], 'span_repr_{}'.format(epoch)))
        torch.save(pairwise_model.state_dict(), os.path.join(config['model_path'], 'pairwise_scorer_{}'.format(epoch)))