import torch
from torch.utils import data
import collections
from itertools import product, combinations
import numpy as np
import jsonlines




class CrossEncoderDataset(data.Dataset):
    def __init__(self, data_path, full_doc=True, multiclass='multiclass', sep_token='</s>', is_training=True):
        super(CrossEncoderDataset, self).__init__()
        with jsonlines.open(data_path, 'r') as f:
            self.data = [topic for topic in f]

        for i, topic in enumerate(self.data):
            self.data[i]['mention_text'] = np.array([' '.join(topic['flatten_tokens'][start:end + 1])
                                                   for start, end, _ in topic['flatten_mentions']])

        self.sep = sep_token
        self.full_doc = full_doc
        if multiclass not in {'coref', 'hypernym', 'multiclass'}:
            raise ValueError(f"The multiclass value needs to be in (coref, hypernym, multiclass), got {multiclass}.")
        self.multiclass = multiclass
        self.is_training = is_training

        self.pairs, self.labels = [], []
        self.info_pairs = []
        for i, topic in enumerate(self.data):
            if self.multiclass == 'multiclass':
                inputs, labels, info_pairs = self.get_topic_pairs(topic)
            elif self.multiclass == 'hypernym':
                inputs, labels, info_pairs = self.get_topic_pair_for_hypernym(topic)
            elif self.multiclass == 'coref':
                inputs, labels, info_pairs = self.get_topic_pairs_for_binary_classification(topic)
            self.pairs.extend(inputs)
            self.labels.extend(labels)
            pair_nums = len(info_pairs)
            info_pairs = np.concatenate((np.array([i] * pair_nums).reshape(pair_nums, 1),
                                        info_pairs), axis=1)
            self.info_pairs.extend(info_pairs)

        if self.multiclass == 'multiclass' or self.multiclass == 'hypernym':
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        else:
            self.labels = torch.tensor(self.labels, dtype=torch.float)


    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]




    def get_topic_pairs_for_binary_classification(self, topic):
        mentions = []
        for mention in topic['mentions']:
            if self.full_doc:
                mentions.append(self.get_full_doc_mention(mention, topic['tokens']))
            else:
                mentions.append(self.get_sentence_context(mention, topic['tokens'], topic['sentences']))
        mentions = np.array(mentions)

        if self.is_training:
            first, second = zip(*combinations(range(len(mentions)), r=2))
        else:
            first, second = zip(*product(range(len(mentions)), repeat=2))

        first, second = np.array(first), np.array(second)
        seps = np.array([self.sep] * len(first))
        inputs = np.char.add(np.char.add(mentions[first], seps), mentions[second]).tolist()


        labels = [topic['mentions'][x][-1] == topic['mentions'][y][-1]
                  for x, y in zip(first, second)]

        return inputs, labels, list(zip(first, second))




    def get_topic_pair_for_hypernym(self, topic):
        '''
                :param topic:
                :return:
                '''
        relations = [(x, y) for x, y in topic['relations']]
        mentions = []
        for mention in topic['mentions']:
            if self.full_doc:
                mentions.append(self.get_full_doc_mention(mention, topic['tokens']))
            else:
                mentions.append(self.get_sentence_context(mention, topic['tokens'], topic['sentences']))
        mentions = np.array(mentions)

        first, second = zip(*[(x, y) for x, y in product(range(len(mentions)), repeat=2) if x != y])
        first, second = np.array(first), np.array(second)
        seps = np.array([self.sep] * len(first))
        inputs = np.char.add(np.char.add(mentions[first], seps), mentions[second]).tolist()

        labels = []
        for x, y in zip(first, second):
            cluster_x, cluster_y = topic['mentions'][x][-1], topic['mentions'][y][-1]
            if (cluster_x, cluster_y) in relations:
                labels.append(1)
            elif (cluster_y, cluster_x) in relations:
                labels.append(2)
            else:
                labels.append(0)

        return inputs, labels, list(zip(first, second))


    def get_topic_pairs(self, topic):
        '''
        :param topic:
        :return:
        '''
        relations = [(x, y) for x, y in topic['relations']]
        mentions = []
        for mention in topic['mentions']:
            if self.full_doc:
                mentions.append(self.get_full_doc_mention(mention, topic['tokens']))
            else:
                mentions.append(self.get_sentence_context(mention, topic['tokens'], topic['sentences']))
        mentions = np.array(mentions)

        first, second = zip(*[(x, y) for x, y in product(range(len(mentions)), repeat=2) if x != y])
        first, second = np.array(first), np.array(second)
        seps = np.array([self.sep] * len(first))
        inputs = np.char.add(np.char.add(mentions[first], seps), mentions[second]).tolist()

        labels = []
        for x, y in zip(first, second):
            cluster_x, cluster_y = topic['mentions'][x][-1], topic['mentions'][y][-1]
            if cluster_x == cluster_y:
                labels.append(1)
            elif (cluster_x, cluster_y) in relations:
                labels.append(2)
            elif (cluster_y, cluster_x) in relations:
                labels.append(3)
            else:
                labels.append(0)

        return inputs, labels, list(zip(first, second))



    def get_full_doc_mention(self, mention, tokens):
        doc_id, start, end, _ = mention
        mention_rep = tokens[doc_id][:start] + ['<m>']
        mention_rep += tokens[doc_id][start:end + 1] + ['</m>']
        mention_rep += tokens[doc_id][end + 1:]
        return ' '.join(mention_rep)




    def get_sentence_context(self, mention, tokens, sentences):
        doc_id, start, end, _ = mention
        sent_start, sent_end = 0, len(tokens) - 1
        i = 0
        while i < len(sentences[doc_id]):
            sent_start, sent_end = sentences[doc_id][i]
            if start >= sent_start and end <= sent_end:
                break
            i += 1

        mention_rep = tokens[doc_id][sent_start:start] + ['<m>']
        mention_rep += tokens[doc_id][start:end + 1] + ['</m>']
        mention_rep += tokens[doc_id][end + 1:sent_end] + [self.sep]

        return ' '.join(mention_rep)




class BiEncoderDataset(data.Dataset):
    def __init__(self, data_path, full_doc=True, multiclass='all', sep_token='</s>',):
        super(BiEncoderDataset, self).__init__()
        with jsonlines.open(data_path, 'r') as f:
            self.data = [topic for topic in f]

        self.full_doc = full_doc
        self.multiclass = multiclass
        self.sep_token = sep_token





class ClusterForHypernymDataset(data.Dataset):
    '''
    generate all pairs of clusters (product)
    where each cluster is the concatenation of the all the mentions
    in the document
    labels:
    0: not related
    1: neutral == hypernym
    2: entailment == hyponym
    '''
    def __init__(self, data_path, with_context=False):
        super(ClusterForHypernymDataset, self).__init__()
        with jsonlines.open(data_path, 'r') as f:
            self.data = [topic for topic in f]
        self.with_context = with_context
        self.premise = []
        self.hypothesis = []
        self.labels = []

        for i, topic in enumerate(self.data):
            self.data[i]['mention_text'] = np.array([' '.join(topic['flatten_tokens'][start:end + 1])
                                                     for start, end, _ in topic['flatten_mentions']])

        for topic_num, topic in enumerate(self.data):
            premise, hypothesis, labels = self.get_topic_candidates(topic)
            self.premise.extend(premise)
            self.hypothesis.extend(hypothesis)
            self.labels.extend(labels)



    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        return self.premise[idx], self.hypothesis[idx], self.labels[idx]



    def get_topic_candidates(self, topic):
        clusters = collections.defaultdict(list)
        for i, (_, _, _, c_id) in enumerate(topic['mentions']):
            clusters[c_id].append(i)

        relations = [(x, y) for x, y in topic['relations']]
        cluster_ids, candidates = [], []
        for c_id, mentions in clusters.items():
            mention_text = topic['mention_text'][mentions]
            sampled = np.random.choice(mention_text, min(len(mention_text), 10), replace=False)
            candidates.append(', '.join(sampled))
            cluster_ids.append(c_id)


        permutations = [(x, y) for x, y in product(range(len(candidates)), repeat=2) if x != y]
        labels = []
        for x, y in permutations:
            if (x, y) in relations:
                labels.append(1)
            elif (y, x) in relations:
                labels.append(2)
            else:
                labels.append(0)
        first, second = zip(*permutations)
        first, second = torch.tensor(first), torch.tensor(second)


        candidates = np.array(candidates)
        premise = candidates[first]
        hypothesis = candidates[second]


        return premise, hypothesis, labels