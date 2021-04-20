import numpy as np
import torch
from itertools import compress, chain, product, combinations
import networkx as nx


def get_pairwise_scores(span_representations, clusters, pairwise_scorer):
    first, second = zip(*list(combinations(range(len(span_representations)), 2)))
    first, second = torch.tensor(first), torch.tensor(second)
    g1, g2 = span_representations[first], span_representations[second]
    labels = clusters[first] == clusters[second]
    with torch.no_grad():
        scores = pairwise_scorer(g1, g2)
        # scores = torch.sigmoid(scores)
    # predictions = (scores > 0.5).to(torch.int)

    return scores, labels



def get_greedy_relations(cluster_ids, scores, first, second):
    graph = nx.DiGraph(directed=True)
    graph.add_nodes_from(cluster_ids)
    _, indices = torch.sort(scores)
    for ind in indices:
        parent, child = cluster_ids[second[ind]], cluster_ids[first[ind]]
        ind = sum([graph.has_edge(node, child) for node in graph.nodes])
        if ind == 0: # to prevent multiple parents to the same node
            graph.add_edge(parent, child)
            if len(list(nx.simple_cycles(graph))) > 0: # to prevent cycle
                graph.remove_edge(parent, child)

    return [[int(a), int(b)] for a, b in graph.edges]


def get_hypernym_relations(topic, clusters, entailment_model, concatenate=True):
    cluster_ids, candidates = [], []
    for c_id, mentions in clusters.items():
        mention_text = topic['mention_text'][mentions]
        sampled = np.random.choice(mention_text, min(len(mention_text), 10), replace=False)
        candidates.append(', '.join(sampled))
        cluster_ids.append(c_id)

    permutations = list(product(range(len(candidates)), repeat=2))
    permutations = [(a, b) for a, b in permutations if a != b]
    first, second = zip(*permutations)
    first, second = torch.tensor(first), torch.tensor(second)


    candidates = np.array(candidates)
    premise = candidates[first]
    hypothesis = candidates[second]

    with torch.no_grad():
        scores, predictions = entailment_model(premise, hypothesis)

    positives = torch.nonzero(predictions == 2).squeeze(-1)
    scores = scores[positives]
    first, second = first[positives], second[positives]

    relations = get_greedy_relations(cluster_ids, scores, first, second)

    return relations
    #
    # graph = nx.DiGraph(directed=False)
    # graph.add_nodes_from(cluster_ids)
    #
    # _, indices = torch.sort(scores, descending=True)
    # for ind in zip(indices):
    #     parent, child = cluster_ids[second[ind]], cluster_ids[first[ind]]
    #
    #     ind = sum([graph.has_edge(node, child) for node in graph.nodes])
    #     if ind == 0: # to prevent multiple parents to the same node
    #         graph.add_edge(parent, child)
    #         if len(list(nx.simple_cycles(graph))) > 0: # to prevent cycle
    #             graph.remove_edge(parent, child)
    #
    # return [[int(a), int(b)] for a, b in graph.edges]






def shorten_large_text(bert_tokens, limit=512):
    return [tokens[:min(limit, len(tokens))] for tokens in bert_tokens]



def pad_and_read_bert(bert_tokens, bert_model):
    doc_ids = list(chain.from_iterable([[i] * len(doc) for i, doc in enumerate(bert_tokens)]))
    # bert_tokens = list(chain.from_iterable(bert_token_ids))
    bert_tokens = shorten_large_text(bert_tokens, limit=512)
    length = np.array([len(d) for d in bert_tokens])
    max_length = max(length)

    if max_length > 512:
        raise ValueError(max_length)

    device = bert_model.device
    docs = torch.tensor([doc + [0] * (max_length - len(doc)) for doc in bert_tokens], device=device)
    attention_masks = torch.tensor([[1] * len(doc) + [0] * (max_length - len(doc)) for doc in bert_tokens],
                                   device=device)
    inputs = {'input_ids': docs, 'attention_mask': attention_masks}

    with torch.no_grad():
        outputs = bert_model(**inputs)

    embeddings = outputs.last_hidden_state



    return doc_ids, embeddings, length





def get_mention_embeddings(topic, docs_embeddings):
    embeddings, width, clusters = [], [], []
    for doc_id, original_start, original_end, cluster_id in topic['mentions']:
        start = topic['starts'][doc_id][original_start]
        end = topic['ends'][doc_id][original_end]

        if start > len(docs_embeddings[doc_id]):
            print(f"problem with topic {topic['id']} and document {doc_id}")
        mention_embedding = docs_embeddings[doc_id][start: end + 1]
        embeddings.append(mention_embedding)
        clusters.append(cluster_id)
        width.append(original_end - original_start)

    return embeddings, width, clusters




def get_all_token_embedding(embedding, start, end):
    span_embeddings, length = [], []
    for s, e in zip(start, end):
        indices = torch.tensor(range(s, e + 1))
        span_embeddings.append(embedding[indices])
        length.append(len(indices))
    return span_embeddings, length