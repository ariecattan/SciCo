from sklearn.cluster import AgglomerativeClustering
import argparse
from transformers import AutoTokenizer, AutoModel
import collections
from tqdm import tqdm
import jsonlines
import os
from datetime import datetime
import pyhocon

from models.baselines import EntailmentModel
from utils.model_utils import *
from utils.corpus import Corpus

import logging
root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)



def get_pairwise_scores(span_representations, clusters, pairwise_scorer):
    first, second = zip(*list(combinations(range(len(span_representations)), 2)))
    first, second = torch.tensor(first), torch.tensor(second)
    labels = clusters[first] == clusters[second]
    g1, g2 = span_representations[first], span_representations[second]
    scores = pairwise_scorer(g1, g2)
    predictions = (scores > 0.5).to(torch.int)

    return predictions, labels



def get_distance_matrix(span_representations, pairwise_scorer):
    num_mentions = len(span_representations)
    first, second = zip(*list(product(range(len(span_representations)), repeat=2)))
    first, second = torch.tensor(first), torch.tensor(second)
    g1, g2 = span_representations[first], span_representations[second]
    scores = pairwise_scorer(g1, g2).detach().cpu()
    distances = 1 - scores.view(num_mentions, num_mentions).numpy()

    return distances







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--data_path', type=str, default='data/data.jsonl')
    parser.add_argument('--bert_model', type=str, default='bert-large-cased')
    parser.add_argument('--output_dir', type=str, default='checkpoints/cosine/bert')
    parser.add_argument('--threshold', type=str, default='0.5')
    args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file(args.config)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info('{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))
    logger.info("pid: {}".format(os.getpid()))

    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device {device}')

    # Load checkpoints and init clustering
    logger.info('Loading checkpoints..')
    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    bert_model = AutoModel.from_pretrained(args.bert_model).to(device)
    hypernym_model = EntailmentModel(config, device)
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                         distance_threshold=float(args.threshold))

    
    logger.info('Loading data..')
    with jsonlines.open(args.data_path, 'r') as f:
        data = [line for line in f]
    data = Corpus(data, tokenizer=bert_tokenizer)

    cosine_similarity = torch.nn.CosineSimilarity()
    doc_ids, starts, ends, cluster_ids = [], [], [], []
    tp, fp, fn = 0, 0, 0

    predicted_data = []

    for topic_num, topic in enumerate(tqdm(data.topics)):
        doc_num, docs_embeddings, docs_length = pad_and_read_bert(topic['bert_tokens'], bert_model)
        continuous_embeddings, width, clusters = get_mention_embeddings(topic, docs_embeddings)
        mention_embeddings = torch.stack([torch.mean(continous, dim=0) for continous in continuous_embeddings])
        clusters = torch.tensor(clusters, device=device)

        # clustering
        distances = get_distance_matrix(mention_embeddings, cosine_similarity)
        predicted = clustering.fit(distances)
        predicted_mentions = np.array(topic['mentions'])
        predicted_clusters = predicted.labels_.reshape(len(predicted_mentions), 1)
        predicted_mentions = np.concatenate((predicted_mentions[:, :-1], predicted_clusters), axis=1)


        all_clusters = collections.defaultdict(list)
        for i, cluster_id in enumerate(predicted.labels_):
            all_clusters[cluster_id].append(i)

        # hypernyms
        if len(all_clusters) > 1:
            relations = get_hypernym_relations(topic, all_clusters, hypernym_model)
        else:
            relations = []

        predicted_data.append({
            "id": topic['id'],
            "tokens": topic['tokens'],
            "mentions": predicted_mentions.tolist(),
            "relations": relations
        })

    logger.info('Saving sys files...')

    jsonl_path = os.path.join(args.output_dir, 'system_{}.jsonl'.format(args.threshold))
    with jsonlines.open(jsonl_path, 'w') as f:
        f.write_all(predicted_data)