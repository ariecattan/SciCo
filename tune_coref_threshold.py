from sklearn.cluster import AgglomerativeClustering
import argparse
import pyhocon
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from utils.conll import write_output_file
from models.baselines import SpanScorer, SimplePairWiseClassifier, SpanEmbedder
from utils.model_utils import *
from utils.utils import *
from predict_cd_coref_entailment import get_distance_matrix
import collections
import jsonlines
from utils.corpus import Corpus
from utils.evaluator import Evaluation

import logging
root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)


def init_models(config, device, model_num):
    span_repr = SpanEmbedder(config, device).to(device)
    span_repr.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                      "span_repr_{}".format(model_num)),
                                         map_location=device))
    span_repr.eval()
    span_scorer = SpanScorer(config).to(device)
    # span_scorer.load_state_dict(torch.load(os.path.join(config['model_path'],
    #                                                     "span_scorer_{}".format(model_num)),
    #                                        map_location=device))
    span_scorer.eval()
    pairwise_scorer = SimplePairWiseClassifier(config).to(device)
    pairwise_scorer.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                           "pairwise_scorer_{}".format(model_num)),
                                              map_location=device))
    pairwise_scorer.eval()

    return span_repr, span_scorer, pairwise_scorer


def predict_topic(topic):
    doc_num, docs_embeddings, docs_length = pad_and_read_bert(topic['bert_tokens'], bert_model)
    continuous_embeddings, width, clusters = get_mention_embeddings(topic, docs_embeddings)


    start_end = torch.stack([torch.cat((mention[0], mention[-1])) for mention in continuous_embeddings])
    width = torch.tensor(width, device=device)


    with torch.no_grad():
        mention_embeddings = span_repr(start_end, continuous_embeddings, width)

    # clustering
    distances = get_distance_matrix(mention_embeddings, pairwise_scorer)

    return distances




def predict_topic_cosine_similarity(topic):
    cosine_similarity = torch.nn.CosineSimilarity()

    doc_num, docs_embeddings, docs_length = pad_and_read_bert(topic['bert_tokens'], bert_model)
    continuous_embeddings, width, clusters = get_mention_embeddings(topic, docs_embeddings)
    mention_embeddings = torch.stack([torch.mean(continous, dim=0) for continous in continuous_embeddings])

    distances = get_distance_matrix(mention_embeddings, cosine_similarity)

    return distances



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_clustering.json')
    parser.add_argument('--cosine', type=str, default='0')
    args = parser.parse_args()

    cosine = True if args.cosine == '1' else False

    config = pyhocon.ConfigFactory.parse_file(args.config)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    logger.info('{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))
    logger.info("pid: {}".format(os.getpid()))


    create_folder(config['save_path'])
    device = 'cuda:{}'.format(config['gpu_num'][0]) if torch.cuda.is_available() else 'cpu'


    logger.info('Loading checkpoints..')
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    config['bert_hidden_size'] = bert_model.config.hidden_size


    logger.info('Loading data..')
    with jsonlines.open(config.data_path, 'r') as f:
        data = [line for line in f]
    data = Corpus(data, tokenizer=bert_tokenizer)


    clustering = []

    for x in np.arange(0.4, 0.65, 0.05):
        agglo = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=x)
        clustering.append(agglo)


    for num in range(5):
        logger.info('Model {}'.format(num))
        if not cosine:
            span_repr, span_scorer, pairwise_scorer = init_models(config, device, num)
        predicted_data = collections.defaultdict(list)

        for topic_num, topic in enumerate(tqdm(data.topics)):


            if cosine:
                distance_matrix = predict_topic_cosine_similarity(topic)
            else:
                distance_matrix = predict_topic(topic)

            for i, agglomerative in enumerate(clustering):

                predicted = agglomerative.fit(distance_matrix)
                predicted_mentions = np.array(topic['mentions'])
                predicted_clusters = predicted.labels_.reshape(len(predicted_mentions), 1)
                predicted_mentions = np.concatenate((predicted_mentions[:, :-1], predicted_clusters), axis=1)
                topic['mentions'] = predicted_mentions.tolist()
            #

                predicted_data[agglomerative.distance_threshold].append({
                    "id": topic['id'],
                    "tokens": topic['tokens'],
                    "mentions": predicted_mentions.tolist(),
                    "relations": []
                })


        for threshold, topics in predicted_data.items():
            doc_name = 'dev_{}_{}_{}'.format(config['linkage_type'], threshold, num)
            write_output_file(topics, dir_path=config['save_path'], doc_name=doc_name)

