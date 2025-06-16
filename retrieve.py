from sentence_transformers import SentenceTransformer
import numpy as np
from index import BM25Index, HNSWIndex


def retrieve_corpus(
    query_scores,
    corpus: list[str],
    args: dict,
    checked_results: dict[str, int] = {},
) -> tuple[list[str], list[float], list[str], list[float]]:
    """
    Retrieve corpus from BM25 and HNSW index
    """
    bm25_index = BM25Index(corpus, args.dataset)
    hnsw_index = HNSWIndex(corpus, args.dataset)
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

    bm25_results, bm25_scores, pos_bm25_scores = bm25_index.search(query_list, args.k)
    if len(neg_queries) > 0:
        _, _, neg_bm25_scores = bm25_index.search(neg_queries, args.k)
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
    # normalize bm25 scores
    bm25_scores = np.array(bm25_scores)
    bm25_scores = (bm25_scores - bm25_scores[-1]) / (
        bm25_scores[0] - bm25_scores[-1] + 1e-6
    )
    hnsw_results, hnsw_scores, query_embs, unique_results, results_embs = (
        hnsw_index.search(query_list, args.k)
    )
    if len(neg_queries) > 0:
        neg_query_embs = SentenceTransformer(hnsw_index.emb_model).encode(
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
    # normalize hnsw scores
    hnsw_scores = np.array(hnsw_scores)
    hnsw_scores = (hnsw_scores - hnsw_scores[-1]) / (
        hnsw_scores[0] - hnsw_scores[-1] + 1e-6
    )
    return bm25_results, bm25_scores, hnsw_results, hnsw_scores


def agg_bm25_results(
    bm25_results: list[list[int]],
    pos_bm25_scores: list[list[float]],
    neg_bm25_scores,
    weights: np.ndarray,
    k: int,
) -> tuple[list[int], list[float]]:
    """
    Aggregate BM25 results from multiple queries
    """
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
    query_embs: np.ndarray,
    unique_results: list[int],
    results_embs: np.ndarray,
    neg_query_emb,
    weights: np.ndarray,
    k: int,
) -> tuple[list[int], list[float]]:
    """
    Aggregate HNSW results from multiple queries
    """
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
