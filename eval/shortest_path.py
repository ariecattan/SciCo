import collections
import networkx as nx
from itertools import product, combinations
from scipy import stats
import numpy as np
from itertools import chain
import sys
import jsonlines


class ShortestPath:
    def __init__(self, gold, system, directed, with_tn):
        self.gold = {x['id']: x for x in gold}
        self.system = {x['id']: x for x in system}
        self.directed = directed
        self.with_tn = with_tn

        self.numerator = []
        self.denominator = []
        self.gold_distances = []
        self.sys_distances = []
        self.all_scores = collections.defaultdict(list)


        for topic in self.gold.keys():
            if topic not in self.system:
                raise ValueError(topic)

            self.topic = topic
            numerator, denominator, gold_scores, sys_scores = \
                self.get_topic_score(self.gold[topic], self.system[topic])

            self.numerator.append(numerator)
            self.denominator.append(denominator)
            self.gold_distances.append(gold_scores)
            self.sys_distances.append(sys_scores)


        self.micro_average = self.compute_micro_average_scores()
        self.macro_average = self.compute_macro_average_score()


    def get_graph(self, topic):
        clusters = [x[-1] for x in topic['mentions']]
        relations = topic['relations']

        graph = nx.DiGraph()
        graph.add_nodes_from(set(clusters))
        graph.add_edges_from(relations)
        return graph

    def get_distance(self, graph, clusters, x, y):
        if clusters[x] == clusters[y]:
            return 1
        elif nx.has_path(graph, clusters[x], clusters[y]):
            return nx.shortest_path_length(graph, clusters[x], clusters[y]) + 1
        else:
            return 0


    def safe_division(self, n, d):
        if n == 0 and d == 0:
            return (1, self.with_tn)
        return (n / d, True)



    def get_topic_score(self, topic_gold, topic_system):
        gold_graph = self.get_graph(topic_gold)
        sys_graph = self.get_graph(topic_system)

        gold_clusters = [x[-1] for x in topic_gold['mentions']]
        sys_clusters = [x[-1] for x in topic_system['mentions']]

        pairs = list(product(range(len(gold_clusters)), repeat=2))
        numerator, denominator = 0, 0

        gold_scores, sys_scores = [], []

        for x, y in pairs:
            gold_distance = self.get_distance(gold_graph, gold_clusters, x, y)
            sys_distance = self.get_distance(sys_graph, sys_clusters, x, y)
            pair_score = self.safe_division(min(gold_distance, sys_distance),
                                            max(gold_distance, sys_distance))

            if pair_score[1]:
                gold_scores.append(gold_distance)
                sys_scores.append(sys_distance)
                numerator += pair_score[0]
                denominator += 1

        self.all_scores[self.topic] = numerator / denominator

        return numerator, denominator, gold_scores, sys_scores



    def compute_micro_average_scores(self):
        return sum(self.numerator) / sum(self.denominator)


    def compute_macro_average_score(self):
        return np.average([ x/y if y != 0 else 0 for x, y in zip(self.numerator, self.denominator)])




if __name__ == '__main__':
    gold_path = sys.argv[1]
    sys_path = sys.argv[2]

    with jsonlines.open(gold_path, 'r') as f:
        gold = [line for line in f]

    with jsonlines.open(sys_path, 'r') as f:
        system = [line for line in f]

    path_ratio = ShortestPath(gold, system, directed=True, with_tn=False)
    print('Path Ratio'.ljust(15),
          'Micro: %.2f' % (path_ratio.micro_average * 100),
          'Macro: %.2f' % (path_ratio.macro_average * 100))