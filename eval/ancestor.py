import collections
import networkx as nx
from itertools import combinations
from scipy import stats
import numpy as np
import sys
import jsonlines
from tqdm import tqdm


class CommonAncestor():
    def __init__(self, gold, system, directed=True):
        self.gold = {x['id']: x for x in gold}
        self.system = {x['id']: x for x in system}
        self.directed = directed

        self.scores = []


        for topic in tqdm(self.gold.keys()):
            if topic not in self.system:
                raise ValueError(topic)

            score = self.get_topic_score(self.gold[topic], self.system[topic])
            self.scores.append(score)

        self.score = np.average(self.scores)


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




    def delta(self, graph, x, y):
        ancestor_x = nx.ancestors(graph, x)
        ancestor_y = nx.ancestors(graph, y)
        ancestor_x.add(x)
        ancestor_y.add(y)

        union = ancestor_x
        union.update(ancestor_y)

        return len(ancestor_x.intersection(ancestor_y)) / len(union)







    def get_topic_score(self, topic_gold, topic_system):
        gold_graph = self.get_graph(topic_gold)
        sys_graph = self.get_graph(topic_system)

        gold_clusters = [x[-1] for x in topic_gold['mentions']]
        sys_clusters = [x[-1] for x in topic_system['mentions']]

        pairs = list(combinations(range(len(gold_clusters)), r=2))
        numerator = 0



        for x, y in pairs:
            x_gold, y_gold = gold_clusters[x], gold_clusters[y]
            x_sys, y_sys = sys_clusters[x], sys_clusters[y]

            distance = abs(self.delta(gold_graph, x_gold, y_gold) -\
                       self.delta(sys_graph, x_sys, y_sys))
            numerator += distance


        return 1 - numerator / len(pairs)



if __name__ == '__main__':
    gold_path = sys.argv[1]
    sys_path = sys.argv[2]

    with jsonlines.open(gold_path, 'r') as f:
        gold = [line for line in f]

    with jsonlines.open(sys_path, 'r') as f:
        system = [line for line in f]

    ancestor = CommonAncestor(gold, system, directed=True)
    print('Common Ancesor'.ljust(15),
          'Avg: %.2f' % (ancestor.score * 100))