import collections
from itertools import chain, product
import networkx as nx
import sys
import jsonlines

class HypernymScore:
    def __init__(self, gold, system):
        self.gold = {x['id']: x for x in gold}
        self.system = {x['id']: x for x in system}

        self.recall_num, self.recall_denominator = [], []
        self.precision_num, self.precision_denominator = [], []
        self.f1 = collections.defaultdict(list)


        for topic in self.gold:
            if topic not in self.system:
                raise ValueError(topic)

            self.topic = topic
            self.compute_individual_hypernym(self.gold[topic], self.system[topic])

        self.micro_recall, self.micro_precision, self.micro_f1 = self.compute_micro_average_scores()




    def get_higher_order_relations(self,  binary_relations):
        graph = nx.DiGraph(directed=True)
        nodes = set(list(chain.from_iterable(binary_relations)))
        graph.add_nodes_from(nodes)
        graph.add_edges_from(binary_relations)

        higher_order_relations = []
        for x, y in product(nodes, repeat=2):
            if x != y and nx.has_path(graph, x, y) and [x, y] not in binary_relations:
                higher_order_relations.append([x, y])

        return higher_order_relations




    def get_expanded_relations(self, binary_relations, mentions):
        expanded_relations = []
        dict_clusters = collections.defaultdict(list)
        for i, mention in enumerate(mentions):
            cluster_id = mention[-1]
            dict_clusters[cluster_id].append(i)

        for parent, child in binary_relations:
            parent_child_relation = []
            for coref_parent in dict_clusters[parent]:
                parent_child_relation.extend([(coref_parent, coref_child) for coref_child in dict_clusters[child]])
            expanded_relations.append(parent_child_relation)

        return expanded_relations






    def compute_individual_hypernym(self, topic_gold, topic_system):
        gold_relations = self.get_expanded_relations(topic_gold['relations'], topic_gold['mentions'])
        system_relations = self.get_expanded_relations(topic_system['relations'], topic_system['mentions'])
        flatten_gold = list(chain.from_iterable(gold_relations))
        flatten_predicted = list(chain.from_iterable(system_relations))

        gold_higher_order_relations = self.get_expanded_relations(self.get_higher_order_relations(topic_gold['relations']), topic_gold['mentions'])
        system_higher_order_relations = self.get_expanded_relations(self.get_higher_order_relations(topic_system['relations']), topic_system['mentions'])



        recall_num, recall_denominator = 0, len(gold_relations)
        precision_num, precision_denominator = 0, len(system_relations)

        for relation in gold_relations:
            score = sum([1 for rel in relation if rel in flatten_predicted + list(chain.from_iterable(system_higher_order_relations))])
            if score > 0:
                recall_num += 1

        for relation in system_relations:
            score = sum([1 for rel in relation if rel in flatten_gold + list(chain.from_iterable(gold_higher_order_relations))])
            if score > 0:
                precision_num += 1

        recall = recall_num / recall_denominator if recall_denominator != 0 else 0
        precision = precision_num / precision_denominator if precision_denominator != 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0
        self.f1[self.topic] = f1

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

    hypernyms = HypernymScore(gold, system)
    print('hypernym'.ljust(15), 'Recall: %.2f' % (hypernyms.micro_recall * 100),
          ' Precision: %.2f' % (hypernyms.micro_precision * 100),
          ' F1: %.2f' % (hypernyms.micro_f1 * 100))