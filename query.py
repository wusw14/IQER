import numpy as np
from constants import DIVERSITY_THRESHOLD
from scipy.stats import ttest_ind


class Query:
    def __init__(self, org_query: str, reformat_template: str):
        self.org_query = org_query
        self.query_scores = {org_query: 2}
        self.queries_from_generated = []
        self.queries_from_table = []
        self.new_queries_from_generated = []
        self.new_queries_from_table = []
        self.query_condition = reformat_template.format(query=org_query)
        self.reformulate_impact = 0
        self.query_list = [org_query]
        self.pos_ids = []
        self.last_query_list = []
        self.best_alpha = 0.5
        self.best_beta = 0.5

    def select_diversified_query_words(self, emb_model, checked_obj_dict: dict):
        query_list = []
        # keep the query words with score > 1
        if np.sum(np.array(list(self.query_scores.values())) > 0) == 1:
            thr = 0
        else:
            thr = 2
        for q, s in self.query_scores.items():
            if s >= thr:
                query_list.append(q)
        selected_query_list = []
        selected_qids = []
        for i, q in enumerate(query_list):
            if q == self.org_query:
                selected_qids.append(i)
                selected_query_list.append(q)
        if len(query_list) == len(selected_query_list):
            self.query_list = selected_query_list
            return selected_query_list
        pos_embs = emb_model.encode(query_list)
        # normalize the pos embs
        pos_embs = pos_embs / np.linalg.norm(pos_embs, axis=1, keepdims=True)
        sim_matrix = np.dot(pos_embs, pos_embs.T)
        threshold = np.median(sim_matrix)
        print(f"Threshold: {threshold}")
        sim_dist = np.round(np.percentile(sim_matrix, list(range(0, 100, 10))), 4)
        print(f"Distribution of similarity scores: {list(sim_dist)}")
        # select the diversified query words from the query list
        sim_scores = np.max(sim_matrix[selected_qids], axis=0)
        # prioritize the newly generated/verified query words
        for i, q in enumerate(query_list):
            if (
                q not in self.new_queries_from_generated
                and q not in self.new_queries_from_table
            ):
                sim_scores[i] = 1
        for i in range(len(query_list) - len(selected_query_list)):
            # select the word with the least similarity score
            qid = np.argmin(sim_scores)
            if sim_scores[qid] > max(DIVERSITY_THRESHOLD, threshold):
                break
            sim_scores = np.maximum(sim_scores, sim_matrix[qid])
            selected_query_list.append(query_list[qid])
            selected_qids.append(qid)
        sim_scores = np.max(sim_matrix[selected_qids], axis=0)
        # select the remaining query words
        for i in range(len(query_list) - len(selected_query_list)):
            # select the word with the least similarity score
            qid = np.argmin(sim_scores)
            if sim_scores[qid] > max(DIVERSITY_THRESHOLD, threshold):
                break
            sim_scores = np.maximum(sim_scores, sim_matrix[qid])
            selected_query_list.append(query_list[qid])
            selected_qids.append(qid)
        self.query_list = selected_query_list
        print(f"Select {len(selected_query_list)} from {len(query_list)}")
        return selected_query_list

    def get_neg_list(self, checked_obj_dict: dict):
        neg_list = []
        for obj, checked in checked_obj_dict.items():
            if not checked:
                neg_list.append(obj)
        # remove the common parts of each item in the neg_list
        pos_word_set = set()
        for q, s in self.query_scores.items():
            if s > 1:
                pos_word_set.update(q.split())
        refined_neg_list = []
        for obj in neg_list:
            obj_words = obj.split()
            obj_words = [word for word in obj_words if word not in pos_word_set]
            if len(obj_words) > 0:
                refined_neg_list.append(" ".join(obj_words))
        if len(refined_neg_list) > 5:
            refined_neg_list = np.random.choice(
                refined_neg_list, 5, replace=False
            ).tolist()
        return refined_neg_list

    def if_reformulate(self, emb_model):
        if len(self.queries_from_table) == 0:
            return True
        # if there exists significant difference between the similarity scores of the query and the retrieved objects
        # --> reformulate
        query_emb = emb_model.encode(self.org_query)
        table_embs = emb_model.encode(self.queries_from_table)
        # normalize the emb
        query_emb = query_emb / np.linalg.norm(query_emb)
        table_embs = table_embs / np.linalg.norm(table_embs, axis=1, keepdims=True)
        query_table_sim_scores = np.dot(query_emb, table_embs.T)
        intra_table_sim_scores = np.dot(table_embs, table_embs.T)
        # flat the sim scores and only keep values less than 1
        query_table_sim_scores = query_table_sim_scores[
            query_table_sim_scores < 1
        ].flatten()
        intra_table_sim_scores = intra_table_sim_scores[
            intra_table_sim_scores < 1
        ].flatten()
        # calculate the t-test p-value
        t_stat, p_value = ttest_ind(query_table_sim_scores, intra_table_sim_scores)
        print(f"t-stat: {t_stat:.4f}, p-value: {p_value:.4f}")
        return p_value < 0.05 and t_stat > 0

    def update_queries_from_generated(self, new_queries_from_generated: list):
        self.new_queries_from_generated = new_queries_from_generated
        self.queries_from_generated.extend(new_queries_from_generated)

    def update_queries_from_table(self, new_queries_from_table: list, corpus: list):
        self.new_queries_from_table = new_queries_from_table
        self.queries_from_table.extend(new_queries_from_table)
        self.pos_ids.extend([corpus.index(q) for q in new_queries_from_table])

    def update_query_scores(self, query_scores: dict):
        self.query_scores.update(query_scores)
