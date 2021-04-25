import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import socket
import yaml
from torch.utils import data
import logging
from datetime import datetime
import os
import torch
from tqdm import tqdm
import jsonlines
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import collections
from itertools import product

from models.datasets import CrossEncoderDataset
from models.muticlass import CorefEntailmentLightning, BinaryCorefLightning, HypernymModel
from utils.model_utils import get_greedy_relations, get_hypernym_relations
from models.baselines import EntailmentModel


class MulticlassInference:
    def __init__(self, dataset, pairwise_scores, coref_threshold, hypernym_threshold):
        self.dataset = dataset
        self.info_pairs = torch.tensor(dataset.info_pairs)
        self.pairwise_scores = pairwise_scores
        self.coref_threshold = coref_threshold
        self.hypernym_threshold = hypernym_threshold


        self.clustering = AgglomerativeClustering(n_clusters=None,
                                                  affinity='precomputed',
                                                  linkage='average',
                                                  distance_threshold=self.coref_threshold)



    def get_coref_adjacency_matrix(self, info_pairs, pairwise_scores):
        '''
        make coreference adjacency matrix
        :param info_pairs:
        :param pairwise_scores:
        :return: adjacency matrix of coref
        '''
        num_of_mentions = info_pairs[:, 1:].max().item() + 1
        adjacency = torch.eye(num_of_mentions)
        coref_predictions = torch.argmax(pairwise_scores, dim=1)
        preds = torch.nonzero(coref_predictions != 0).squeeze(-1)
        pairs = info_pairs[preds][:, 1:]
        coref_scores = pairwise_scores[preds][:, 1]
        adjacency.index_put_(tuple(pairs.t()), coref_scores)

        return adjacency.numpy()


    def undirect_adjacency_matrix(self, matrix):
        undirected = matrix.copy()
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                maxi = max(matrix[i, j], matrix[j, i])
                undirected[i, j] = maxi
                undirected[j, i] = maxi

        return undirected

    def get_mention_pairs(self, cluster_a, cluster_b, pairs, scores):
        cluster_pairs = [[x, y] for x in cluster_a for y in cluster_b]
        indices = [i for i, x in enumerate(pairs) if x in cluster_pairs]
        return scores[indices].mean()


    def hypernym_relations(self, predicted_mentions, pairs, scores):
        '''
        return relations between clusters hypernym scores=2, hyponym=3
        :param clusters:
        :param pairs:
        :param scores:
        :return:
        '''

        clusters = collections.defaultdict(list)
        for i, (_, _, _, c_id) in enumerate(predicted_mentions):
            clusters[c_id].append(i)

        cluster_ids = list(clusters.keys())
        permutations = list(product(range(len(cluster_ids)), repeat=2))
        permutations = [(x, y) for x, y in permutations if x != y]
        first, second = zip(*permutations)
        info_pairs = pairs[:, 1:].tolist()
        avg_scores = torch.stack([self.get_mention_pairs(clusters[cluster_ids[x]], clusters[cluster_ids[y]], info_pairs, scores[:, 3])
                                  for x, y in zip(first, second)])

        inds = torch.nonzero(avg_scores >= self.hypernym_threshold).squeeze(-1)
        avg_scores = avg_scores[inds]
        first, second = torch.tensor(first)[inds], torch.tensor(second)[inds]

        relations = get_greedy_relations(cluster_ids, avg_scores, first, second)

        return relations




    def get_topic_prediction(self, topic_num, topic_pairs, topic_scores):
        # coref clusters
        coref_adjacency_matrix = self.get_coref_adjacency_matrix(topic_pairs, topic_scores)
        undirected_matrix = self.undirect_adjacency_matrix(coref_adjacency_matrix)
        distance_matrix = 1 - undirected_matrix
        predicted = self.clustering.fit(distance_matrix)


        mentions = self.dataset.data[topic_num]['mentions']
        mentions = np.array(mentions)
        predicted_clusters = predicted.labels_.reshape(len(mentions), 1)
        predicted_mentions = np.concatenate((mentions[:, :-1], predicted_clusters), axis=1)


        relations = self.hypernym_relations(predicted_mentions, topic_pairs, topic_scores)

        return {
            "id": self.dataset.data[topic_num]['id'],
            "tokens":self.dataset.data[topic_num]['tokens'],
            "mentions": predicted_mentions.tolist(),
            "relations": relations
        }


    def predict_cluster_relations(self):
        idx, vals = torch.unique(self.info_pairs[:, 0], return_counts=True)
        all_pairs = torch.split_with_sizes(self.info_pairs, tuple(vals))
        all_scores = torch.split_with_sizes(self.pairwise_scores, tuple(vals))

        predicted_data = []
        for topic, (topic_pair, topic_scores) in enumerate(tqdm(zip(all_pairs, all_scores), total=len(all_scores))):
            # if topic == 44:
            #     print('BUGGG')
            data = self.get_topic_prediction(topic, topic_pair, topic_scores)
            predicted_data.append(data)

        self.predicted_data = predicted_data


    def save_predicted_file(self, output_dir):
        jsonl_path = os.path.join(output_dir, 'system_{}_{}.jsonl'.format(self.coref_threshold, self.hypernym_threshold))
        with jsonlines.open(jsonl_path, 'w') as f:
            f.write_all(self.predicted_data)



class BinaryCoreferenceInference:
    def __init__(self, dataset, pairwise_scores, threshold):
        self.dataset = dataset
        self.info_pairs = torch.tensor(dataset.info_pairs)
        self.pairwise_scores = pairwise_scores
        self.threshold = threshold

        self.clustering = AgglomerativeClustering(n_clusters=None,
                                                  affinity='precomputed',
                                                  linkage='average',
                                                  distance_threshold=self.threshold)




    def get_topic_prediction(self, topic_num, topic_scores):
        mentions = self.dataset.data[topic_num]['mentions']
        num_mentions = len(mentions)
        distance_matrix = 1 - topic_scores.view(num_mentions, num_mentions).cpu().numpy()
        predicted = self.clustering.fit(distance_matrix)

        mentions = np.array(mentions)
        predicted_clusters = predicted.labels_.reshape(len(mentions), 1)
        predicted_mentions = np.concatenate((mentions[:, :-1], predicted_clusters), axis=1)

        return {
            "id": self.dataset.data[topic_num]['id'],
            "tokens": self.dataset.data[topic_num]['tokens'],
            "mentions": predicted_mentions.tolist(),
            "relations": []
        }



    def fit(self):
        idx, vals = torch.unique(self.info_pairs[:, 0], return_counts=True)
        all_pairs = torch.split_with_sizes(self.info_pairs, tuple(vals))
        all_scores = torch.split_with_sizes(self.pairwise_scores, tuple(vals))

        predicted_data = []
        for topic, (topic_pair, topic_scores) in enumerate(tqdm(zip(all_pairs, all_scores), total=len(all_scores))):
            data = self.get_topic_prediction(topic, topic_scores)
            predicted_data.append(data)

        self.predicted_data = predicted_data



class EntailmentInference:
    def __init__(self, dataset, predicted_data, entailment_model):
        self.dataset = dataset
        self.entailment_model = entailment_model
        self.predicted_data = []

        for topic_num, data in enumerate(tqdm(predicted_data)):
            topic = self.dataset.data[topic_num]
            mentions = predicted_data[topic_num]['mentions']
            relations = self.get_topic_relations(topic, mentions)

            topic_data = data.copy()
            topic_data['relations'] = relations
            self.predicted_data.append(topic_data)



    def get_topic_relations(self, topic, data):
        clusters = collections.defaultdict(list)
        for i, (_, _, _, c_id) in enumerate(data):
            clusters[c_id].append(i)

        relations = get_hypernym_relations(topic, clusters, self.entailment_model)

        return relations



class HypernymInference:
    def __init__(self, dataset, predicted_data, pairwise_scores, threshold):
        self.dataset = dataset
        self.info_pairs = torch.tensor(dataset.info_pairs)
        self.pairwise_scores = pairwise_scores
        self.predicted_data = predicted_data
        self.hypernym_threshold = threshold


    def get_mention_pairs(self, cluster_a, cluster_b, pairs, scores):
        cluster_pairs = [[x, y] for x in cluster_a for y in cluster_b]
        indices = [i for i, x in enumerate(pairs) if x in cluster_pairs]
        return scores[indices].mean()


    def get_topic_relations(self, pairs, scores, predicted_data):
        predicted_mentions = predicted_data['mentions']
        clusters = collections.defaultdict(list)
        for i, (_, _, _, c_id) in enumerate(predicted_mentions):
            clusters[c_id].append(i)

        cluster_ids = list(clusters.keys())
        permutations = list(product(range(len(cluster_ids)), repeat=2))
        permutations = [(x, y) for x, y in permutations if x != y]
        first, second = zip(*permutations)
        info_pairs = pairs[:, 1:].tolist()

        avg_scores = torch.stack([self.get_mention_pairs(clusters[cluster_ids[x]], clusters[cluster_ids[y]], info_pairs, scores[:, 2])
                                  for x, y in zip(first, second)])

        inds = torch.nonzero(avg_scores >= self.hypernym_threshold).squeeze(-1)
        avg_scores = avg_scores[inds]
        first, second = torch.tensor(first)[inds], torch.tensor(second)[inds]

        relations = get_greedy_relations(cluster_ids, avg_scores, first, second)

        return relations


    def fit(self):
        idx, vals = torch.unique(self.info_pairs[:, 0], return_counts=True)
        all_pairs = torch.split_with_sizes(self.info_pairs, tuple(vals))
        all_scores = torch.split_with_sizes(self.pairwise_scores, tuple(vals))

        all_predicted_data = []
        for topic, (topic_pair, topic_score, predicted_data) in enumerate(tqdm(zip(all_pairs, all_scores, self.predicted_data),
                                                                            total=len(all_pairs))):
            relations = self.get_topic_relations(topic_pair, topic_score, predicted_data)
            data_with_rel = predicted_data.copy()
            data_with_rel['relations'] = relations
            all_predicted_data.append(data_with_rel)

        self.predicted_data = all_predicted_data




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/multiclass.yaml')
    parser.add_argument('--multiclass', type=str, default='pipeline')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_name = args.multiclass

    if model_name not in {'pipeline', 'multiclass'}:
        raise ValueError(f"The multiclass value needs to be in (multiclass, pipeline), got {model_name}.")

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
    logger.info("pid: {}".format(os.getpid()))
    logger.info('Server name: {}'.format(socket.gethostname()))

    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])

    logger.info(f"Using {model_name} model")
    logger.info('loading models')
    pl_logger = CSVLogger(save_dir='logs', name='multiclass_inference')
    trainer = pl.Trainer(gpus=config['gpu_num'], accelerator='dp', logger=pl_logger)



    #### multiclass
    if model_name == 'multiclass':
        logger.info('Predicting multiclass scores')
        model = CorefEntailmentLightning.load_from_checkpoint(config['checkpoint_multiclass'], config=config)
        test = CrossEncoderDataset(config["data"]["test_set"],
                                   full_doc=config['full_doc'],
                                   multiclass=model_name,
                                   is_training=False)
        test_loader = data.DataLoader(test,
                                      batch_size=config["model"]["batch_size"] * 64,
                                      shuffle=False,
                                      collate_fn=model.tokenize_batch,
                                      num_workers=16,
                                      pin_memory=True)
        results = trainer.predict(model, dataloaders=test_loader)
        results = torch.cat([torch.tensor(x) for x in results])
        torch.save(results, 'checkpoints/multiclass/multiclass_results.pt')
        # results = torch.load('test_multiclass_results.pt')
        inference = MulticlassInference(test, results, config['agg_threshold'], config['hypernym_threshold'])
        inference.predict_cluster_relations()
        inference.save_predicted_file(config['save_path'])

    else:
        coref_model = BinaryCorefLightning.load_from_checkpoint(config['checkpoint_coref'], config=config)
        hypernym_model = HypernymModel.load_from_checkpoint(config['checkpoint_hypernym'], config=config)
        test_coref = CrossEncoderDataset(config["data"]["test_set"],
                                   full_doc=config['full_doc'],
                                   multiclass='coref',
                                   is_training=False)
        test_coref_loader = data.DataLoader(test_coref,
                                            batch_size=config['model']['batch_size'] * 64,
                                            shuffle=False,
                                            collate_fn=coref_model.tokenize_batch,
                                            num_workers=16,
                                            pin_memory=True)
        test_hypernym = CrossEncoderDataset(config["data"]["test_set"],
                                         full_doc=config['full_doc'],
                                         multiclass='hypernym',
                                         is_training=False)
        test_hypernym_loader = data.DataLoader(test_hypernym,
                                            batch_size=config['model']['batch_size'] * 32,
                                            shuffle=False,
                                            collate_fn=hypernym_model.tokenize_batch,
                                            num_workers=16,
                                            pin_memory=True)

        logger.info('Predicting coreference scores')
        coref_results = trainer.predict(coref_model, dataloaders=test_coref_loader)
        coref_results = torch.cat([torch.tensor(x) for x in coref_results])
        torch.save(coref_results, 'checkpoints/test_cross_encoder_coref.pt')
        # coref_results = torch.load('checkpoints/coref_results.pt')
        coref_inference = BinaryCoreferenceInference(test_coref, coref_results, config['agg_threshold'])
        coref_inference.fit()
        predicted_data = coref_inference.predicted_data

        logger.info('Predicting hypernym scores')
        hypernym_results = trainer.predict(hypernym_model, dataloaders=test_hypernym_loader)
        hypernym_results = torch.cat([torch.tensor(x) for x in hypernym_results])
        torch.save(hypernym_results, 'checkpoints/test_cross_encoder_hypernym.pt')
        # hypernym_results = torch.load('checkpoints/hypernym_results.pt')
        hypernym_inference = HypernymInference(test_hypernym, predicted_data, hypernym_results, config['hypernym_threshold'])
        hypernym_inference.fit()
        predicted_data = hypernym_inference.predicted_data


        jsonl_path = os.path.join(config['save_path'], 'system_{}_{}.jsonl'.format(config['agg_threshold'],
                                                                                   config['hypernym_threshold']))
        with jsonlines.open(jsonl_path, 'w') as f:
            f.write_all(predicted_data)




        # logger.info('Predicting relations between clusters')
        # entailment_model = EntailmentModel(config['nli_model'], device=config['gpu_num'][0])
        # entailment_inference = EntailmentInference(test, data, entailment_model=entailment_model)
        #
        # jsonl_path = os.path.join(config['save_path'], 'system_{}.jsonl'.format(config['agg_threshold']))
        # with jsonlines.open(jsonl_path, 'w') as f:
        #     f.write_all(entailment_inference.predicted_data)