import collections
from itertools import chain, product
import networkx as nx
import sys
import jsonlines

class HypernymScore50:
    def __init__(self, gold, system):
        self.gold = {x['id']: x for x in gold}
        self.system = {x['id']: x for x in system}

        self.recall_num, self.recall_denominator = [], []
        self.precision_num, self.precision_denominator = [], []
        self.recall, self.precision, self.f1 = {}, {}, {}


        for topic in self.gold:
            if topic not in self.system:
                raise ValueError(topic)

            self.topic = topic
            self.get_topic_score(self.gold[topic], self.system[topic])

        self.micro_recall, self.micro_precision, self.micro_f1 = self.compute_micro_average_scores()
        self.macro_f1 = sum(self.f1.values()) / len(self.f1)



    def get_higher_order_relations(self,  binary_relations):
        graph = nx.DiGraph(directed=True)
        nodes = set(list(chain.from_iterable(binary_relations)))
        graph.add_nodes_from(nodes)
        graph.add_edges_from(binary_relations)

        higher_order_relations = []
        for x, y in product(nodes, repeat=2):
            if x != y and nx.has_path(graph, x, y) and [x, y] not in binary_relations:
                higher_order_relations.append((x, y))

        return higher_order_relations


    def get_clusters(self, topic):
        clusters = collections.defaultdict(list)
        for i, (_, _, _, c_id) in enumerate(topic['mentions']):
            clusters[c_id].append(i)

        return clusters


    def get_candidate_clusters(self, parent, children, clusters):
        candidate_parents, candidate_children = [], []
        for cluster_id, mentions in clusters.items():
            intersect_x = [x for x in parent if x in mentions]
            intersect_y = [x for x in children if x in mentions]

            if len(intersect_x) >= 0.5 * len(parent):
                candidate_parents.append(cluster_id)
            if len(intersect_y) >= 0.5 * len(children):
                candidate_children.append(cluster_id)

        return candidate_parents, candidate_children



    def get_topic_score(self, topic_gold, topic_system):
        gold_relations = [(x, y) for x, y in topic_gold['relations']]
        gold_relations.extend(self.get_higher_order_relations(gold_relations))
        gold_relations = set(gold_relations)
        gold_clusters = self.get_clusters(topic_gold)


        sys_relations = [(x, y) for x, y in topic_system['relations']]
        sys_relations.extend(self.get_higher_order_relations(sys_relations))
        sys_relations = set(sys_relations)
        sys_clusters = self.get_clusters(topic_system)


        recall_num, precision_num = 0, 0
        recall_denominator, precision_denominator = len(gold_relations), len(sys_relations)

        # recall
        for x, y in gold_relations:
            parent, children = gold_clusters[x], gold_clusters[y]
            candidate_parents, candidate_children = self.get_candidate_clusters(parent, children, sys_clusters)
            potentials = [[cand_par, cand_child] for cand_par in candidate_parents
                          for cand_child in candidate_children]

            if sum([1 for x, y in potentials if (x, y) in sys_relations]) >= 1:
                recall_num += 1


        # precision
        for x, y in sys_relations:
            parent, children = sys_clusters[x], sys_clusters[y]
            candidate_parents, candidate_children = self.get_candidate_clusters(parent, children, gold_clusters)
            potentials = [[cand_par, cand_child] for cand_par in candidate_parents
                          for cand_child in candidate_children]

            if sum([1 for x, y in potentials if (x, y) in gold_relations]) >= 1:
                precision_num += 1


        recall = recall_num / recall_denominator if recall_denominator != 0 else 0
        precision = precision_num / precision_denominator if precision_denominator != 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0
        self.f1[self.topic] = f1
        self.recall[self.topic] = recall
        self.precision[self.topic] = precision

        self.recall_num.append(recall_num)
        self.recall_denominator.append(recall_denominator)
        self.precision_num.append(precision_num)
        self.precision_denominator.append(precision_denominator)






    def compute_micro_average_scores(self):
        micro_recall = sum(self.recall_num) / sum(self.recall_denominator) \
            if sum(self.recall_denominator) != 0 else 0
        micro_precision = sum(self.precision_num) / sum(self.precision_denominator) \
            if sum(self.precision_denominator) != 0 else 0
        micro_f1 = 2 * micro_recall * micro_precision / (micro_recall + micro_precision) \
            if (micro_recall + micro_precision) != 0 else 0

        return micro_recall, micro_precision, micro_f1


if __name__ == '__main__':
    gold_path = sys.argv[1]
    sys_path = sys.argv[2]

    with jsonlines.open(gold_path, 'r') as f:
        gold = [line for line in f]

    with jsonlines.open(sys_path, 'r') as f:
        system = [line for line in f]

    hypernyms = HypernymScore50(gold, system)
    print('Hypernym 50%'.ljust(15), 'Recall: %.2f' % (hypernyms.micro_recall * 100),
          ' Precision: %.2f' % (hypernyms.micro_precision * 100),
          ' F1: %.2f' % (hypernyms.micro_f1 * 100))