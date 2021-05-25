from sklearn.cluster import AgglomerativeClustering
import argparse
import pyhocon
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import collections
import jsonlines

from models.baselines import SpanScorer, SimplePairWiseClassifier, SpanEmbedder, EntailmentModel
from utils.utils import *


from utils.model_utils import *
from utils.corpus import Corpus
from utils.evaluator import Evaluation

import logging
root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)





def init_models(config, device):
    span_repr = SpanEmbedder(config, device).to(device)
    span_repr.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                      "span_repr_{}".format(config['model_num'])),
                                         map_location=device))
    span_repr.eval()

    pairwise_scorer = SimplePairWiseClassifier(config).to(device)
    pairwise_scorer.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                           "pairwise_scorer_{}".format(config['model_num'])),
                                              map_location=device))
    pairwise_scorer.eval()

    return span_repr, pairwise_scorer






def get_distance_matrix(span_representations, pairwise_scorer):
    num_mentions = len(span_representations)
    first, second = zip(*list(product(range(len(span_representations)), repeat=2)))
    first, second = torch.tensor(first), torch.tensor(second)
    g1, g2 = span_representations[first], span_representations[second]

    scores = pairwise_scorer(g1, g2).detach().cpu()
    distances = 1 - scores.view(num_mentions, num_mentions).numpy()

    return distances





def predict_topic(topic, hypernym_model=None):
    doc_num, docs_embeddings, docs_length = pad_and_read_bert(topic['bert_tokens'], bert_model)
    continuous_embeddings, width, clusters = get_mention_embeddings(topic, docs_embeddings)
    start_end = torch.stack([torch.cat((mention[0], mention[-1])) for mention in continuous_embeddings])
    width = torch.tensor(width, device=device)
    clusters = torch.tensor(clusters, device=device)

    with torch.no_grad():
        mention_embeddings = span_repr(start_end, continuous_embeddings, width)

    # clustering
    distances = get_distance_matrix(mention_embeddings, pairwise_scorer)
    predicted = clustering.fit(distances)
    predicted_mentions = np.array(topic['mentions'])
    predicted_clusters = predicted.labels_.reshape(len(predicted_mentions), 1)
    predicted_mentions = np.concatenate((predicted_mentions[:, :-1], predicted_clusters), axis=1)
    topic['mentions'] = predicted_mentions.tolist()
    #
    #


    relations = []
    if hypernym_model is not None:
        all_clusters = collections.defaultdict(list)
        for i, cluster_id in enumerate(predicted.labels_):
            all_clusters[cluster_id].append(i)

        if len(np.unique(predicted.labels_)) > 1:
            relations = get_hypernym_relations(topic, all_clusters, hypernym_model)


    predicted_data.append({
        "id": topic['id'],
        "tokens": topic['tokens'],
        "mentions": predicted_mentions.tolist(),
        "relations": relations
    })

    return predicted_data






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_clustering.json')
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

    device = 'cuda:{}'.format(config['gpu_num'][0]) if torch.cuda.is_available() else 'cpu'


    # Load checkpoints and init clustering
    logger.info('Loading checkpoints..')
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    hypernym_model = EntailmentModel(config['nli_model'], device)

    config['bert_hidden_size'] = bert_model.config.hidden_size
    span_repr, pairwise_scorer = init_models(config, device)
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=config['threshold'])

    logger.info('Loading data..')
    with jsonlines.open(config.data_path, 'r') as f:
        data = [line for line in f]
    data = Corpus(data, tokenizer=bert_tokenizer)


    doc_ids, starts, ends, cluster_ids = [], [], [], []
    tp, fp, fn = 0, 0, 0

    predicted_data = []
    logger.info('Inference')

    all_scores, all_labels = [], []
    for topic_num, topic in enumerate(tqdm(data.topics)):
        doc_num, docs_embeddings, docs_length = pad_and_read_bert(topic['bert_tokens'], bert_model)
        continuous_embeddings, width, clusters = get_mention_embeddings(topic, docs_embeddings)
        start_end = torch.stack([torch.cat((mention[0], mention[-1])) for mention in continuous_embeddings])
        width = torch.tensor(width, device=device)
        clusters = torch.tensor(clusters, device=device)

        with torch.no_grad():
            mention_embeddings = span_repr(start_end, continuous_embeddings, width)

        # pairwise classification
        pairwise_predictions, pairwise_labels = get_pairwise_scores(mention_embeddings, clusters, pairwise_scorer)
        all_scores.extend(pairwise_predictions.squeeze(1))
        all_labels.extend(pairwise_labels.to(torch.int))


        # clustering
        distances = get_distance_matrix(mention_embeddings, pairwise_scorer)
        predicted = clustering.fit(distances)

        predicted_mentions = np.array(topic['mentions'])
        predicted_clusters = predicted.labels_.reshape(len(predicted_mentions), 1)
        predicted_mentions = np.concatenate((predicted_mentions[:, :-1], predicted_clusters), axis=1)
        topic['mentions'] = predicted_mentions.tolist()


        all_clusters = collections.defaultdict(list)
        for i, cluster_id in enumerate(predicted.labels_):
            all_clusters[cluster_id].append(i)


        # hierarchial relations
        if len(np.unique(predicted.labels_)) > 1:
            relations = get_hypernym_relations(topic, all_clusters, hypernym_model)
        else:
            relations = []

        predicted_data.append({
            "id": topic['id'],
            "tokens": topic['tokens'],
            "mentions": predicted_mentions.tolist(),
            "relations": relations
        })



    all_scores = torch.stack(all_scores)
    all_labels = torch.stack(all_labels)
    strict_preds = (all_scores > 0).to(torch.int)
    eval = Evaluation(strict_preds, all_labels.to(device))
    logger.info('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
    logger.info('Number of positive pairs: {}/{}'.format(len(torch.nonzero(all_labels == 1)),
                                                         len(all_labels)))
    logger.info('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                    eval.get_precision(), eval.get_f1()))


    logger.info('Saving sys files...')
    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])
    jsonl_path = os.path.join(config['save_path'], 'system_{}.jsonl'.format(config['threshold']))
    with jsonlines.open(jsonl_path, 'w') as f:
        f.write_all(predicted_data)