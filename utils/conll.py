import collections
import operator
import os
import networkx as nx
import numpy as np

def output_conll(data, doc_word_map, doc_start_map, doc_end_map):
    predicted_conll = []

    flatten_index = 0
    for doc_id, doc in enumerate(data['tokens']):
        for token_id, token in enumerate(doc):
            token_key = '{}_{}'.format(doc_id, token_id)
            if token_key in doc_word_map:
                clusters = '|'.join(['({})'.format(x) for x in doc_word_map[token_key]])
            elif token_key in doc_start_map:
                clusters = '|'.join(['({}'.format(x) for x in doc_start_map[token_key]])
            elif token_key in doc_end_map:
                clusters = '|'.join(['{})'.format(x) for x in doc_end_map[token_key]])
            else:
                clusters = '-'

            predicted_conll.append([flatten_index, doc_id, token_id, token, clusters])
            flatten_index += 1

    return predicted_conll




def get_dict_map(predicted_mentions):
    doc_start_map = collections.defaultdict(list)
    doc_end_map = collections.defaultdict(list)
    doc_word_map = collections.defaultdict(list)

    for doc_id, start, end, cluster_id in predicted_mentions:
        start_key = '{}_{}'.format(doc_id, start)
        end_key = '{}_{}'.format(doc_id, end)

        if start == end:
            doc_word_map[start_key].append(cluster_id)
        else:
            doc_start_map[start_key].append((cluster_id, end))
            doc_end_map[end_key].append((cluster_id, start))

    for k, v in doc_start_map.items():
        doc_start_map[k] = [cluster_id for cluster_id, end_key in sorted(v, key=operator.itemgetter(1), reverse=True)]
    for k, v in doc_end_map.items():
        doc_end_map[k] = [cluster_id for cluster_id, end_key in sorted(v, key=operator.itemgetter(1), reverse=True)]


    return doc_start_map, doc_end_map, doc_word_map




def get_connected_clusters(prediction):
    graph = nx.Graph()

    mentions = np.array(prediction['mentions'])

    graph.add_nodes_from(mentions[:, -1])
    graph.add_edges_from(prediction['relations'])
    connected = [g for g in nx.connected_components(graph)]

    map_cluster_id = {}
    for i, cluster_list in enumerate(connected):
        for x in cluster_list:
            map_cluster_id[x] = i

    clusters = [int(x) for x in mentions[:, -1]]
    new_clusters = np.array([map_cluster_id[x] for x in clusters]).reshape(len(clusters), 1)

    mentions = np.concatenate((mentions[:, :-1], new_clusters), axis=1)

    return mentions




def write_connected_components(predictions, dir_path, doc_name):
    output_path = os.path.join(dir_path, '{}_connected.conll'.format(doc_name))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(output_path, 'w') as f:
        for prediction in predictions:
            mentions = get_connected_clusters(prediction)
            # prediction['mention'] = mentions

            doc_start_map, doc_end_map, doc_word_map = get_dict_map(mentions)
            conll_tokens = output_conll(prediction, doc_word_map, doc_start_map, doc_end_map)

            f.write('#begin document {}\n'.format(prediction['id']))
            for token in conll_tokens:
                f.write('\t'.join([str(x) for x in token]) + '\n')
            f.write('#end document\n')





def write_output_file(predictions, dir_path, doc_name):
    output_path = os.path.join(dir_path, '{}_simple.conll'.format(doc_name))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(output_path, 'w') as f:
        for prediction in predictions:
            doc_start_map, doc_end_map, doc_word_map = get_dict_map(prediction['mentions'])
            conll_tokens = output_conll(prediction, doc_word_map, doc_start_map, doc_end_map)

            f.write('#begin document {}\n'.format(prediction['id']))
            for token in conll_tokens:
                f.write('\t'.join([str(x) for x in token]) + '\n')
            f.write('#end document\n')



def get_surrounding_context(doc_sentences):
    context = [[x - 1, x + 1] for x in doc_sentences]

    starts, ends = [], []
    starts.append(context[0][0])
    i = 1

    while i < len(doc_sentences):
        if context[i][0] > context[i - 1][1] + 2:
            ends.append(context[i - 1][1])
            starts.append(context[i][0])
        i += 1
    ends.append(context[-1][1])

    return list(zip(starts, ends))