from sentence_transformers import SentenceTransformer
import numpy as np
from index import BM25Index, HNSWIndex
from pydantic import BaseModel
from typing import Optional
import time
from utils import cal_ndcg


class RetrievedInfo(BaseModel):
    bm25_queries: list[str]
    hnsw_queries: list[str]
    bm25_objs: list[str]
    bm25_agg_pos_scores: list[float]
    bm25_agg_neg_scores: list[float]
    bm25_agg_pos_scores_avg: list[float]
    bm25_agg_pos_scores_max: list[float]
    bm25_ids: list[list[int]]
    bm25_scores: list[list[float]]
    bm25_unique_ids: list[int]
    bm25_pos_scores: list[list[float]]
    bm25_neg_scores: Optional[list[list[float]]] = None
    hnsw_objs: list[str]
    hnsw_agg_pos_scores: list[float]
    hnsw_agg_neg_scores: list[float]
    hnsw_agg_pos_scores_avg: list[float]
    hnsw_agg_pos_scores_max: list[float]
    hnsw_ids: list[list[int]]
    hnsw_scores: list[list[float]]
    hnsw_unique_ids: list[int]
    hnsw_pos_scores: list[list[float]]
    hnsw_neg_scores: Optional[list[list[float]]] = None


def retrieve_corpus(
    bm25_queries: list[str],
    hnsw_queries: list[str],
    corpus: list[str],
    args: dict,
    bm25_index: BM25Index,
    hnsw_index: HNSWIndex,
    pos_ids: list[int] = [],
    neg_ids: list[int] = [],
) -> RetrievedInfo:
    """
    Retrieve corpus from BM25 and HNSW index
    """
    start_time = time.time()
    bm25_ids, bm25_scores, unique_bm25_ids, pos_bm25_scores = bm25_index.search(
        bm25_queries, args.k
    )
    # print(f"Time for BM25 retrieval: {time.time() - start_time:.4f}s")
    neg_queries = [corpus[i] for i in neg_ids]
    neg_bm25_scores = bm25_index.get_neg_scores(neg_queries, unique_bm25_ids)
    # TODO: normalize the pos/neg scores
    (
        bm25_agg_ids,
        bm25_agg_pos_scores,
        bm25_agg_neg_scores,
        bm25_agg_pos_scores_avg,
        bm25_agg_pos_scores_max,
    ) = agg_results(
        unique_bm25_ids,
        pos_bm25_scores,
        neg_bm25_scores,
        args.k,
    )
    # print(f"Time for BM25 aggregation: {time.time() - start_time:.4f}s")
    bm25_objs = [corpus[i] for i in bm25_agg_ids]
    # print(f"Time for BM25 objects: {time.time() - start_time:.4f}s")

    hnsw_ids, hnsw_scores, query_embs, unique_hsnw_ids, results_embs = (
        hnsw_index.search(hnsw_queries, args.k)
    )
    # print(f"Time for HNSW retrieval: {time.time() - start_time:.4f}s")
    pos_hnsw_scores = np.dot(query_embs, results_embs.T)
    if len(neg_queries) > 0:
        neg_query_embs = hnsw_index.emb_model.encode(neg_queries, batch_size=512)
        # normalize the query embeddings
        neg_query_embs = neg_query_embs / np.linalg.norm(
            neg_query_embs, axis=1, keepdims=True
        )
        neg_hnsw_scores = np.dot(neg_query_embs, results_embs.T)
    else:
        neg_hnsw_scores = None
    (
        hnsw_agg_ids,
        hnsw_agg_pos_scores,
        hnsw_agg_neg_scores,
        hnsw_agg_pos_scores_avg,
        hnsw_agg_pos_scores_max,
    ) = agg_results(
        unique_hsnw_ids,
        pos_hnsw_scores,
        neg_hnsw_scores,
        args.k,
    )
    # print(f"Time for HNSW aggregation: {time.time() - start_time:.4f}s")
    hnsw_objs = [corpus[i] for i in hnsw_agg_ids]
    retrieved_info = RetrievedInfo(
        bm25_queries=bm25_queries,
        hnsw_queries=hnsw_queries,
        bm25_objs=bm25_objs,
        bm25_agg_pos_scores=bm25_agg_pos_scores,
        bm25_agg_neg_scores=bm25_agg_neg_scores,
        bm25_agg_pos_scores_avg=bm25_agg_pos_scores_avg,
        bm25_agg_pos_scores_max=bm25_agg_pos_scores_max,
        bm25_ids=bm25_ids,
        bm25_scores=bm25_scores,
        bm25_unique_ids=unique_bm25_ids,
        bm25_pos_scores=pos_bm25_scores,
        bm25_neg_scores=neg_bm25_scores,
        hnsw_objs=hnsw_objs,
        hnsw_agg_pos_scores=hnsw_agg_pos_scores,
        hnsw_agg_neg_scores=hnsw_agg_neg_scores,
        hnsw_agg_pos_scores_avg=hnsw_agg_pos_scores_avg,
        hnsw_agg_pos_scores_max=hnsw_agg_pos_scores_max,
        hnsw_ids=hnsw_ids,
        hnsw_scores=hnsw_scores,
        hnsw_unique_ids=unique_hsnw_ids,
        hnsw_pos_scores=pos_hnsw_scores,
        hnsw_neg_scores=neg_hnsw_scores,
    )
    return retrieved_info


def combine_max_avg(pos_scores_avg, pos_scores_max, pos_ids, retrieved_ids):
    if len(pos_ids) > 0:
        best_beta = 0.5
        best_score = 0
        for i in range(0, 11):
            beta = i / 10
            scores = beta * pos_scores_avg + (1 - beta) * pos_scores_max
            sorted_ids, _ = zip(
                *sorted(zip(retrieved_ids, scores), key=lambda x: x[1], reverse=True)
            )
            ndcg_score = cal_ndcg(sorted_ids, pos_ids)
            if ndcg_score > best_score or (ndcg_score == best_score and beta == 0.5):
                best_score = ndcg_score
                best_beta = beta
        print(f"Best beta (beta * avg + (1 - beta) * max): {best_beta:.1f}")
        return best_beta * pos_scores_avg + (1 - best_beta) * pos_scores_max
    else:
        return 0.5 * pos_scores_avg + 0.5 * pos_scores_max


def choose_best_param(pos_scores, neg_scores, retrieved_ids, pos_ids) -> float:
    best_param = 0
    best_score = -1
    for param in range(0, 11):
        scores = pos_scores - neg_scores * param / 10
        sorted_ids, _ = zip(
            *sorted(zip(retrieved_ids, scores), key=lambda x: x[1], reverse=True)
        )
        ndcg_score = cal_ndcg(sorted_ids, pos_ids)
        if ndcg_score > best_score:
            best_score = ndcg_score
            best_param = param
    print(f"Best parameter: {best_param / 10:.1f}")
    return best_param / 10


def agg_results(
    unique_ids: list[int],
    pos_scores: list[list[float]],
    neg_scores: list[list[float]],
    k: int,
) -> tuple[list[int], list[float], list[float]]:
    """
    Aggregate BM25 results from multiple queries
    """
    pos_scores = np.array(pos_scores)
    # min_scalar = np.min(pos_scores)
    pos_scores_avg = np.mean(pos_scores, axis=0)
    pos_scores_max = np.max(pos_scores, axis=0)
    agg_pos_scores = (pos_scores_avg + pos_scores_max) / 2
    # pct = 100 - 5 / len(unique_ids)
    # pos_scores = combine_max_avg(pos_scores_avg, pos_scores_max, pos_ids, unique_ids)
    if neg_scores is not None:
        agg_neg_scores = np.max(neg_scores, axis=0)
    else:
        agg_neg_scores = np.zeros_like(agg_pos_scores)
        neg_scores = np.zeros_like(pos_scores)
    retrieved_ids, agg_pos_scores, agg_neg_scores, pos_scores_avg, pos_scores_max = zip(
        *sorted(
            zip(
                unique_ids,
                agg_pos_scores,
                agg_neg_scores,
                pos_scores_avg,
                pos_scores_max,
            ),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    min_scalar = max(np.min(agg_pos_scores), 0)
    max_scalar = max(np.max(agg_pos_scores), 1e-6)
    # k = min(k, np.sum(np.array(pos_scores_max) > 0))
    k = np.sum(np.array(pos_scores_max) > 0)
    print(f"retrieved num [k]: {k}")
    retrieved_ids = retrieved_ids[:k]
    agg_pos_scores = agg_pos_scores[:k]
    agg_neg_scores = agg_neg_scores[:k]
    pos_scores_avg = pos_scores_avg[:k]
    pos_scores_max = pos_scores_max[:k]
    # relative rank
    # scores = [1 / np.log2(i + 2) if s > 0 else 0 for i, s in enumerate(scores)]
    # print(f"[DEBUG] min_scalar: {min_scalar:.4f}, max_scalar: {max_scalar:.4f}")
    # score_distribution = np.percentile(scores, np.arange(0, 101, 10))
    # print(f"[DEBUG] scores: {list(np.round(score_distribution, 4))}")
    agg_pos_scores = (np.array(agg_pos_scores) - min_scalar) / (max_scalar - min_scalar)
    agg_neg_scores = (np.array(agg_neg_scores) - min_scalar) / (max_scalar - min_scalar)
    agg_pos_scores = np.maximum(agg_pos_scores, 0)
    agg_neg_scores = np.maximum(agg_neg_scores, 0)
    # pos_scores = (np.array(pos_scores) - min_scalar) / (max_scalar - min_scalar)
    # neg_scores = (np.array(neg_scores) - min_scalar) / (max_scalar - min_scalar)
    # pos_scores = np.maximum(pos_scores, 0)
    # neg_scores = np.maximum(neg_scores, 0)
    agg_pos_scores = agg_pos_scores.tolist()
    agg_neg_scores = agg_neg_scores.tolist()
    # pos_scores = pos_scores.tolist()
    # neg_scores = neg_scores.tolist()
    print(f"min_scalar: {min_scalar:.4f}, max_scalar: {max_scalar:.4f}")
    return retrieved_ids, agg_pos_scores, agg_neg_scores, pos_scores_avg, pos_scores_max
