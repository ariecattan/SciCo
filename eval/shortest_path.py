import collections
import networkx as nx
from itertools import product
from scipy import stats
import numpy as np
from itertools import chain



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
        # self.all_scores = self.compute_all_scores()


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

        if self.directed:
            graph = nx.DiGraph(directed=True)
        else:
            graph = nx.Graph()

        graph.add_nodes_from(list(range(len(topic['mentions']))))
        graph.add_edges_from(edges)

        return graph


    def get_distance(self, graph, clusters, x, y):
        if clusters[x] == clusters[y]:
            return 1
        elif nx.has_path(graph, x, y):
            return nx.shortest_path_length(graph, x, y) + 1
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


    def compute_all_scores(self):
        return [x / y for x, y in zip(self.numerator, self.denominator)]

    def compute_macro_average_score(self):
        return np.average([ x/y if y != 0 else 0 for x, y in zip(self.numerator, self.denominator)])



    #
    #
    # def get_all_scores(gold, system, directed, with_tn=False):
    #     overall_num, overall_denum = 0, 0
    #     overall_gold_scores, overall_sys_scores = [], []
    #     spearman_avg = []
    #     macro_avg = 0
    #     for topic in gold.keys():
    #         if topic not in system:
    #             raise ValueError(topic)
    #
    #         numerator, denominator, gold_scores, sys_scores = get_overall_score_v2(gold[topic], system[topic],
    #                                                                                directed=directed, with_tn=with_tn)
    #         overall_num += numerator
    #         overall_denum += denominator
    #
    #         sperman_topic = stats.spearmanr(gold_scores, sys_scores)
    #         spearman_avg.append(sperman_topic[0])
    #
    #         overall_gold_scores.extend(gold_scores)
    #         overall_sys_scores.extend(sys_scores)
    #
    #         macro_avg += numerator / denominator
    #
    #     micro_avg = round(overall_num / overall_denum * 100, 1)
    #     macro_avg = round(macro_avg / len(gold) * 100, 1)
    #     spearman = stats.spearmanr(overall_sys_scores, overall_gold_scores)
    #
    #     sparman_macro = round(sum(spearman_avg) / len(spearman_avg) * 100, 1)
    #     return micro_avg, macro_avg, sparman_macro, (round(spearman[0] * 100, 1), spearman[1])
    #
