import jsonlines
import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm

from utils.corpus import Corpus
from utils.model_utils import *
from models.baselines import EntailmentModel
from utils.conll import write_output_file, write_connected_components

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)



def singleton_baseline(data, hypernym_model):
    predicted_data = []

    for topic_num, topic in enumerate(tqdm(data.topics)):
        predicted_mentions = np.array(topic['mentions'])
        predicted_clusters = np.array(range(len(topic['mentions']))).reshape(len(predicted_mentions), 1)
        predicted_mentions = np.concatenate((predicted_mentions[:, :-1],
                                             predicted_clusters), axis=1)
        all_clusters = {i: [i] for i in range(len(topic['mentions']))}

        relations = get_hypernym_relations(topic, all_clusters, hypernym_model)
        predicted_data.append({
            "id": topic['id'],
            "tokens": topic['tokens'],
            "mentions": predicted_mentions.tolist(),
            "relations": relations
        })

    return predicted_data




def single_cluster(data):
    predicted_data = []
    for topic_num, topic in enumerate(tqdm(data.topics)):
        predicted_mentions = np.array(topic['mentions'])
        predicted_clusters = np.array([0] * len(topic['mentions'])).reshape(len(topic['mentions']), 1)
        predicted_mentions = np.concatenate((predicted_mentions[:, :-1],
                                             predicted_clusters), axis=1)

        predicted_data.append({
            "id": topic['id'],
            "tokens": topic['tokens'],
            "mentions": predicted_mentions.tolist(),
            "relations": []
        })

    return predicted_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--data_path', type=str, default='data/test/jsonl/binu.jsonl')
    parser.add_argument('--output_dir', type=str, default='checkpoints/boundary')
    parser.add_argument('--nli_model', type=str, default='roberta-large-mnli', help='Entailment model for the relations')
    args = parser.parse_args()


    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info(args)
    logger.info('{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))
    logger.info("pid: {}".format(os.getpid()))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info('Loading data..')
    with jsonlines.open(args.data_path, 'r') as f:
        raw_data = [line for line in f]

    data = Corpus(raw_data, tokenizer=None)
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    hypernym_model = EntailmentModel(args.nli_model, device)

    logger.info('Getting singleton baseline')
    singleton_data = singleton_baseline(data, hypernym_model)
    with jsonlines.open(os.path.join(args.output_dir, 'singleton_baseline.jsonl'), 'w') as f:
        f.write_all(singleton_data)
    write_output_file(singleton_data, args.output_dir, 'singleton')
    write_connected_components(singleton_data, args.output_dir, 'singleton')

    logger.info('Getting single cluster baseline')
    single_cluster_data = single_cluster(data)
    with jsonlines.open(os.path.join(args.output_dir, 'all.jsonl'), 'w') as f:
        f.write_all(single_cluster_data)

    write_output_file(single_cluster_data, args.output_dir, 'all')
    write_connected_components(single_cluster_data, args.output_dir, 'all')
