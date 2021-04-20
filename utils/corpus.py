import numpy as np
import collections
import torch




class Corpus:
    def __init__(self, data, tokenizer):
        self.topics = data
        self.tokenizer = tokenizer
        self.segment_window = 512

        self.topic_bert_ids = []
        self.topic_starts = []
        self.topic_ends = []




        # self.flatten_relation = self.get_flatten_relations(data['mentions'], data['relations'])

        for i, topic in enumerate(self.topics):
            self.topics[i]['mention_text'] = np.array([' '.join(topic['flatten_tokens'][start:end + 1])
                                                       for start, end, _ in topic['flatten_mentions']])
            self.topics[i]['id'] = data[i]['id']

            if self.tokenizer is not None:
                bert_tokens, starts, ends = self.tokenize_topic(topic)

                self.topics[i]['bert_tokens'] = bert_tokens
                self.topics[i]['starts'] = starts
                self.topics[i]['ends'] = ends







    def get_expanded_relations(self, binary_relations, mentions):
        expanded_relations = []

        clusters = collections.defaultdict(list)
        for i, (_, _, _, cluster_id) in enumerate(mentions):
            clusters[cluster_id].append(i)
        mention2cluster = {i: clusters[cluster_id] for i, (_, _, _, cluster_id)
                           in enumerate(mentions)}

        for parent, child in binary_relations:
            parent_child_relation = []
            for coref_parent in mention2cluster[parent]:
                parent_child_relation.extend([(coref_parent, coref_child) for coref_child in mention2cluster[child]])
            expanded_relations.append(parent_child_relation)

        return expanded_relations







    def tokenize_topic(self, topic):
        topic_bert_tokens, topic_starts, topic_ends = [], [], []
        for doc in topic['tokens']:
            bert_tokens, starts, ends = self.tokenize_doc(doc)
            topic_bert_tokens.append(bert_tokens)
            topic_starts.append(starts)
            topic_ends.append(ends)

        return topic_bert_tokens, topic_starts, topic_ends



    def split_doc_into_segments(self, bert_tokens, sentence_ids, with_special_tokens=True):
        segments = [0]
        current_token = 0
        max_segment_length = self.segment_window
        if with_special_tokens:
            max_segment_length -= 2
        while current_token < len(bert_tokens):
            end_token = min(len(bert_tokens) - 1, current_token + max_segment_length - 1)
            sentence_end = sentence_ids[end_token]
            if end_token != len(bert_tokens) - 1 and sentence_ids[end_token + 1] == sentence_end:
                while end_token >= current_token and sentence_ids[end_token] == sentence_end:
                    end_token -= 1

                if end_token < current_token:
                    raise ValueError(bert_tokens)

            current_token = end_token + 1
            segments.append(current_token)

        return segments


    def tokenize_doc(self, paragraph):
        bert_tokens_ids = [0]
        start_bert_idx, end_bert_idx = [], []
        alignment = []
        bert_cursor = 0
        for i, token in enumerate(paragraph):
            bert_token = self.tokenizer.encode(token)[1:-1]
            if bert_token:
                bert_tokens_ids.extend(bert_token)
                bert_start_index = bert_cursor + 1
                start_bert_idx.append(bert_start_index)
                bert_cursor += len(bert_token)
                bert_end_index = bert_cursor
                end_bert_idx.append(bert_end_index)
                alignment.extend([i] * len(bert_token))

        bert_tokens_ids.append(2)
        return bert_tokens_ids, start_bert_idx, end_bert_idx


