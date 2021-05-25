import jsonlines
from sklearn.cluster import AgglomerativeClustering
import textdistance
from multiprocessing import Pool
import numpy as np
import sys

clustering = AgglomerativeClustering(n_clusters=None,
                                     affinity='precomputed',
                                     distance_threshold=0.3,
                                     linkage='average')


def get_topic_cluster(topic):
    mentions = [' '.join(topic['flatten_tokens'][start:end+1])
                for start, end, cluster_id in topic['flatten_mentions']]
    similarity = [[textdistance.levenshtein.normalized_similarity(x, y)
                   for x in mentions] for y in mentions]

    distances = 1 - np.array(similarity)

    predicted = clustering.fit(distances)
    predicted_mentions = np.array(topic['mentions'])
    predicted_clusters = predicted.labels_.reshape(len(predicted_mentions), 1)
    predicted_mentions = np.concatenate((predicted_mentions[:, :-1], predicted_clusters), axis=1)

    predictions = {
        "tokens": topic['tokens'],
        "mentions": predicted_mentions.tolist(),
        "relations": [],
        "id": topic['id']
    }

    return predictions


if __name__ == '__main__':
    data_path = sys.argv[1]
    output = sys.argv[2]

    with jsonlines.open(data_path, 'r') as f:
        data = [line for line in f]

    with Pool(processes=40) as p:
        predicted_data = p.map(get_topic_cluster, data)

    with jsonlines.open(output, 'w') as f:
        f.write_all(predicted_data)