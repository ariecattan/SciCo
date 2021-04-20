from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import chain, product
import collections
import networkx as nx
import sys
import jsonlines


class AUC:
    def __init__(self, gold, system):
        self.gold = {x['id']: x for x in gold}
        self.system = {x['id']: x for x in system}
        self.all_scores = []

        self.gold_pred = []
        self.sys_pred = []
        self.all_scores = collections.defaultdict(list)

        for topic in self.gold.keys():
            if topic not in self.system:
                raise ValueError(f"The topic {topic['id']} does noe exist in the system")

            print(topic)
            gold_topic_pred = self.get_multiclass_prediction(self.gold[topic])
            sys_topic_pred = self.get_multiclass_prediction(self.system[topic])

            self.gold_pred.extend(gold_topic_pred)
            self.sys_pred.extend(sys_topic_pred)

            if topic == 'Image Model Blocks 1' or topic == 'Transformers 0':
                continue

            y_hat_topic = self.get_one_hot_vector(sys_topic_pred)
            topic_score = roc_auc_score(gold_topic_pred, y_hat_topic, multi_class='ovo', average='macro')
            self.all_scores[topic] = topic_score



        # y = self.get_one_hot_vector(self.gold_pred)
        y_hat = self.get_one_hot_vector(self.sys_pred)
        self.score = roc_auc_score(self.gold_pred, y_hat, multi_class='ovo', average='macro')


    def get_one_hot_vector(self, vector):
        one_hot = np.zeros((len(vector), 4))
        vector = np.array(vector)
        one_hot[np.arange(vector.size), vector] = 1
        return one_hot


    def get_graph(self, topic):
        clusters = [x[-1] for x in topic['mentions']]
        relations = topic['relations']

        cluster_dic = collections.defaultdict(list)
        for i, cluster in enumerate(clusters):
            cluster_dic[cluster].append(i)

        edges = []
        for x, y in relations:
            parents = cluster_dic[x]
            children = cluster_dic[y]
            edges.extend([(p, c) for p in parents for c in children])


        graph = nx.DiGraph(directed=True)
        graph.add_nodes_from(list(range(len(topic['mentions']))))
        graph.add_edges_from(edges)

        return graph




    def get_multiclass_prediction(self, topic):
        '''
        0 not related, 1 coref, 2 hypernym, 3 hyponym
        :param topic:
        :return:
        '''
        graph = self.get_graph(topic)
        clusters = [x[-1] for x in topic['mentions']]

        predictions = []
        for x, y in product(range(len(clusters)), repeat=2):
            if clusters[x] == clusters[y]:
                predictions.append(1)
            elif nx.has_path(graph, x, y):
                predictions.append(2)
            elif nx.has_path(graph, y, x):
                predictions.append(3)
            else:
                predictions.append(0)

        return predictions




if __name__ == '__main__':
    gold_path = sys.argv[1]
    sys_path = sys.argv[2]

    with jsonlines.open(gold_path, 'r') as f:
        gold = [line for line in f]

    with jsonlines.open(sys_path, 'r') as f:
        system = [line for line in f]

    roc = AUC(gold, system)
    print(f'AUC score: {roc.score}')