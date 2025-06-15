from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import hnswlib
from sentence_transformers import SentenceTransformer

# from colbert.infra import Run, RunConfig, ColBERTConfig
# from colbert import Indexer
# from colbert import Searcher
# from colbert.data import Queries
import os
from load_data import load_data
import numpy as np
import json
import sys
import time
from collections import defaultdict
import numpy as np


def hnsw_query(corpus: list, query: list, dataset_name: str, top_n=10):
    # encode query
    query = [q.lower() for q in query]
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    query_embedding = SentenceTransformer(emb_model).encode(query, batch_size=512)
    emb_dim = query_embedding.shape[1]
    # load HNSW index
    index_path = f"index/{dataset_name}_hnsw_index.bin"
    p = hnswlib.Index(space="cosine", dim=emb_dim)  # assuming 768-dim embeddings
    p.load_index(index_path)
    # search for nearest neighbors
    labels, distances = p.knn_query(query_embedding, k=top_n)
    results = []
    scores = []
    unique_results = []
    for i, q in enumerate(query):
        result, score = [], []
        for j in range(top_n):
            doc_id = int(labels[i][j])
            result.append(doc_id)
            score.append(1 - distances[i][j])
        results.append(result)
        scores.append(score)
        unique_results.extend(result)
    unique_results = list(set(unique_results))
    # get the embeddings of the unique results
    results_embs = p.get_items(unique_results, return_type="numpy")
    return results, scores, query_embedding, unique_results, results_embs


def bm25_query(corpus: list, query: list, dataset_name: str, top_n=10):
    # load BM25 index
    with open(f"index/{dataset_name}_bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    stop_words = set(stopwords.words("english"))
    results = []
    scores = []
    org_scores = []
    for q in query:
        # tokenization
        tokenized_query = [
            token for token in word_tokenize(q.lower()) if token not in stop_words
        ]
        score = bm25.get_scores(tokenized_query)
        org_scores.append(score)
        result = np.argsort(score)[::-1][:top_n].tolist()  # get top_n indices
        score = np.array(score)[result]
        results.append(result)
        scores.append(score)
    return results, scores, org_scores


# def colbert_query(corpus: list, query: list, dataset_name: str, top_n=10):
#     # transform the query to tsv format if not exists
#     # if not os.path.exists(f"data/{dataset_name}_query.tsv"):
#     with open(f"data/{dataset_name}_query.tsv", "w") as f:
#         for i, q in enumerate(query):
#             f.write(f"{i}\t{q}\n")
#     with Run().context(RunConfig(nranks=1, experiment=dataset_name)):
#         config = ColBERTConfig(nbits=2, root=dataset_name)
#         searcher = Searcher(index=dataset_name, config=config)
#         queries = Queries(f"data/{dataset_name}_query.tsv")
#         ranking = searcher.search_all(queries, k=top_n)
#     ranking = ranking.data
#     results = []
#     scores = []
#     for i, q in enumerate(query):
#         result = []
#         score = []
#         for j in range(min(top_n, len(ranking[i]))):
#             doc_id = ranking[i][j][0]
#             result.append(doc_id)
#             score.append(ranking[i][j][2])
#         results.append(result)
#         scores.append(score)
#     return results, scores


def agg_results(results, scores, k):
    obj_score_sum, obj_score_max = {}, {}
    for res, score in zip(results, scores):
        for obj, s in zip(res, score):
            if obj not in obj_score_sum:
                obj_score_sum[obj] = 0
                obj_score_max[obj] = -1e6
            obj_score_sum[obj] += s
            obj_score_max[obj] = max(obj_score_max[obj], s)
    obj_score = {}
    num = len(results)
    for obj, s_sum in obj_score_sum.items():
        obj_score[obj] = s_sum / num + obj_score_max[obj]
    # get top k objects
    sorted_obj_score = sorted(obj_score.items(), key=lambda x: x[1], reverse=True)
    sorted_objs = [obj for obj, _ in sorted_obj_score[:k]]
    sorted_scores = [score for _, score in sorted_obj_score[:k]]
    return sorted_objs, sorted_scores


def retrieve_corpus(query_scores, corpus, args, checked_results={}):
    if type(query_scores) == list:
        query_list = list(query_scores)
        weights = [1.0 / len(query_list) for _ in query_list]
        weights = np.array(weights).reshape(-1, 1)
    elif type(query_scores) == dict:
        query_list = list(query_scores.keys())
        weights = [query_scores[q] for q in query_list]
        weights = np.array(weights) / sum(weights)
        weights = weights.reshape(-1, 1)
    else:
        raise ValueError(f"Invalid query_scores type: {type(query_scores)}")
    if len(checked_results) > 0:
        neg_queries = [q for q, v in checked_results.items() if v == 0]
        if len(neg_queries) > 5:
            neg_queries = np.random.choice(neg_queries, 5)
    else:
        neg_queries = []

    bm25_results, bm25_scores, pos_bm25_scores = bm25_query(
        corpus, query_list, args.dataset, args.k
    )
    if len(neg_queries) > 0:
        _, _, neg_bm25_scores = bm25_query(corpus, neg_queries, args.dataset, args.k)
    else:
        neg_bm25_scores = None
    bm25_results, bm25_scores = agg_bm25_results(
        bm25_results,
        pos_bm25_scores,
        neg_bm25_scores,
        weights,
        args.k,
    )
    bm25_results = [corpus[i] for i in bm25_results]
    hnsw_results, hnsw_scores, query_embs, unique_results, results_embs = hnsw_query(
        corpus, query_list, args.dataset, args.k
    )
    if len(neg_queries) > 0:
        emb_model = "sentence-transformers/all-MiniLM-L6-v2"
        neg_query_embs = SentenceTransformer(emb_model).encode(
            neg_queries, batch_size=512
        )
        # normalize the query embeddings
        neg_query_embs = neg_query_embs / np.linalg.norm(
            neg_query_embs, axis=1, keepdims=True
        )
        neg_query_emb = neg_query_embs.mean(axis=0)
    else:
        neg_query_emb = None
    hnsw_results, hnsw_scores = agg_hnsw_results(
        query_embs, unique_results, results_embs, neg_query_emb, weights, args.k
    )
    hnsw_results = [corpus[i] for i in hnsw_results]
    return bm25_results, bm25_scores, hnsw_results, hnsw_scores


def agg_bm25_results(bm25_results, pos_bm25_scores, neg_bm25_scores, weights, k):
    retrieved_ids = []
    for res in bm25_results:
        retrieved_ids.extend(res)
    retrieved_ids = list(set(retrieved_ids))
    pos_bm25_scores = np.array(pos_bm25_scores)
    pos_bm25_scores = pos_bm25_scores[:, retrieved_ids]
    pos_bm25_scores = pos_bm25_scores * weights
    pos_scores = pos_bm25_scores.sum(axis=0)
    if neg_bm25_scores is not None:
        neg_bm25_scores = np.array(neg_bm25_scores)
        neg_bm25_scores = neg_bm25_scores[:, retrieved_ids]
        neg_scores = np.mean(neg_bm25_scores, axis=0)
        scores = pos_scores - neg_scores
    else:
        scores = pos_scores
    retrieved_ids, scores = zip(
        *sorted(zip(retrieved_ids, scores), key=lambda x: x[1], reverse=True)
    )
    retrieved_ids = retrieved_ids[:k]
    scores = scores[:k]
    return retrieved_ids, scores


def agg_hnsw_results(
    query_embs, unique_results, results_embs, neg_query_emb, weights, k
):
    # normalize the query embeddings
    query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    # calculate the similarity between the query and the results
    pos_scores = np.dot(query_embs, results_embs.T)
    pos_scores = pos_scores * weights
    pos_scores = pos_scores.sum(axis=0)
    if neg_query_emb is not None:
        neg_scores = np.dot(neg_query_emb, results_embs.T)
        scores = pos_scores - neg_scores
    else:
        scores = pos_scores
    retrieved_ids, scores = zip(
        *sorted(zip(unique_results, scores), key=lambda x: x[1], reverse=True)
    )
    retrieved_ids = retrieved_ids[:k]
    scores = scores[:k]
    return retrieved_ids, scores
