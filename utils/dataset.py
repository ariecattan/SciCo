import torch
from torch.utils import data
import collections
from itertools import product
import numpy as np
import pytorch_lightning as pl


class HypernymDataset(data.Dataset):
    def __init__(self, corpus, tokenizer):
        self.tokenizer = tokenizer
        self.premise = []
        self.hypothesis = []
        self.labels = []

        for topic_num, topic in enumerate(corpus.topics):
            premise, hypothesis, labels = self.get_topic_candidates(topic)
            self.premise.extend(premise)
            self.hypothesis.extend(hypothesis)
            self.labels.extend(labels)


        self.premise = np.array(self.premise)
        self.hypothesis = np.array(self.hypothesis)
        self.tokens = self.concat_premise_hypothesis(self.premise, self.hypothesis)
        self.labels = torch.tensor(self.labels, dtype=torch.long)


    def concat_premise_hypothesis(self, premise, hypothesis):
        seps = np.array(['</s></s>'] * len(premise))
        inputs = np.char.add(np.char.add(premise, seps), hypothesis).tolist()
        tokens = self.tokenizer(inputs, padding=True)

        return tokens


    def get_topic_candidates(self, topic):
        mention_text = topic['mention_text']
        mentions = topic['mentions']

        clusters = collections.defaultdict(list)
        for i, m in enumerate(mentions):
            cluster_id = m[-1]
            clusters[cluster_id].append(mention_text[i])

        cluster_ids, candidates = [], []
        for c_id, mentions in clusters.items():
            sampled = np.random.choice(mentions, min(len(mentions), 10), replace=False)
            candidates.append(', '.join(sampled))
            cluster_ids.append(c_id)

        permutations = list(product(range(len(candidates)), repeat=2))
        permutations = [(a, b) for a, b in permutations if a != b]
        labels = [2 if [b, a] in topic['relations'] else 1 for a, b in permutations]
        first, second = zip(*permutations)
        first = torch.tensor(first)
        second = torch.tensor(second)

        candidates = np.array(candidates)
        premise = candidates[first]
        hypothesis = candidates[second]

        return premise, hypothesis, labels



    def __getitem__(self, idx):
        # return self.premise[idx], self.hypothesis[idx], self.labels[idx]
        # return self.tokens[idx], self.labels[idx]
        item = {k: torch.tensor(val[idx]) for k, val in self.tokens.items()}
        item['labels'] = self.labels[idx]
        return item, self.labels[idx]


    def __len__(self):
        return len(self.labels)





class HypernymWithContextDataset(data.Dataset):
    def __init__(self, corpus):
        self.corpus = corpus







