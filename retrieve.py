from sentence_transformers import SentenceTransformer
import numpy as np
from index import BM25Index, HNSWIndex
from pydantic import BaseModel
from query import Query
from typing import Optional
import time


class RetrievedInfo(BaseModel):
    query_list: list[str]
    bm25_objs: list[str]
    bm25_agg_scores: list[float]
    bm25_ids: list[list[int]]
    bm25_scores: list[list[float]]
    bm25_unique_ids: list[int]
    bm25_pos_scores: list[list[float]]
    bm25_neg_scores: Optional[list[list[float]]] = None
    hnsw_objs: list[str]
    hnsw_agg_scores: list[float]
    hnsw_ids: list[list[int]]
    hnsw_scores: list[list[float]]
    hnsw_unique_ids: list[int]
    hnsw_pos_scores: list[list[float]]
    hnsw_neg_scores: Optional[list[list[float]]] = None


def retrieve_corpus(
    query_list: list[str],
    corpus: list[str],
    args: dict,
    bm25_index: BM25Index,
    hnsw_index: HNSWIndex,
    neg_queries: list[str] = [],
    pos_ids: list[int] = [],
) -> RetrievedInfo:
    """
    Retrieve corpus from BM25 and HNSW index
    """
    start_time = time.time()
    bm25_ids, bm25_scores, unique_bm25_ids, pos_bm25_scores = bm25_index.search(
        query_list, args.k
    )
    # print(f"Time for BM25 retrieval: {time.time() - start_time:.4f}s")
    neg_bm25_scores = bm25_index.get_neg_scores(neg_queries, unique_bm25_ids)
    bm25_agg_ids, bm25_agg_scores = agg_results(
        unique_bm25_ids,
        pos_bm25_scores,
        neg_bm25_scores,
        args.k,
        pos_ids,
    )
    # print(f"Time for BM25 aggregation: {time.time() - start_time:.4f}s")
    bm25_objs = [corpus[i] for i in bm25_agg_ids]
    # print(f"Time for BM25 objects: {time.time() - start_time:.4f}s")

    hnsw_ids, hnsw_scores, query_embs, unique_hsnw_ids, results_embs = (
        hnsw_index.search(query_list, args.k)
    )
    # print(f"Time for HNSW retrieval: {time.time() - start_time:.4f}s")
    pos_hnsw_scores = np.dot(query_embs, results_embs.T)
    if len(neg_queries) > 0:
        neg_query_embs = SentenceTransformer(hnsw_index.emb_model).encode(
            neg_queries, batch_size=512
        )
        # normalize the query embeddings
        neg_query_embs = neg_query_embs / np.linalg.norm(
            neg_query_embs, axis=1, keepdims=True
        )
        neg_query_embs = neg_query_embs.mean(axis=0, keepdims=True)
        neg_hnsw_scores = np.dot(neg_query_embs, results_embs.T)
    else:
        neg_hnsw_scores = None
    hnsw_agg_ids, hnsw_agg_scores = agg_results(
        unique_hsnw_ids,
        pos_hnsw_scores,
        neg_hnsw_scores,
        args.k,
        pos_ids,
    )
    # print(f"Time for HNSW aggregation: {time.time() - start_time:.4f}s")
    hnsw_objs = [corpus[i] for i in hnsw_agg_ids]
    retrieved_info = RetrievedInfo(
        query_list=query_list,
        bm25_objs=bm25_objs,
        bm25_agg_scores=bm25_agg_scores,
        bm25_ids=bm25_ids,
        bm25_scores=bm25_scores,
        bm25_unique_ids=unique_bm25_ids,
        bm25_pos_scores=pos_bm25_scores,
        bm25_neg_scores=neg_bm25_scores,
        hnsw_objs=hnsw_objs,
        hnsw_agg_scores=hnsw_agg_scores,
        hnsw_ids=hnsw_ids,
        hnsw_scores=hnsw_scores,
        hnsw_unique_ids=unique_hsnw_ids,
        hnsw_pos_scores=pos_hnsw_scores,
        hnsw_neg_scores=neg_hnsw_scores,
    )
    return retrieved_info


def if_combine_max_avg(pos_scores_avg, pos_scores_max, pos_ids, retrieved_ids) -> bool:
    if len(pos_ids) > 0:
        sorted_ids, _ = zip(
            *sorted(
                zip(retrieved_ids, pos_scores_avg), key=lambda x: x[1], reverse=True
            )
        )
        avg_score = cal_ndcg(sorted_ids, pos_ids)
        sorted_ids, _ = zip(
            *sorted(
                zip(retrieved_ids, pos_scores_max + pos_scores_avg),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        avg_max_score = cal_ndcg(sorted_ids, pos_ids)
        return avg_score < avg_max_score
    else:
        return False


def cal_ndcg(sorted_ids, pos_ids) -> float:
    dcg = 0
    for i, id in enumerate(sorted_ids):
        if id in pos_ids:
            dcg += 1 / np.log2(i + 2)
    return dcg


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
    unique_objs: list[int],
    pos_scores: list[list[float]],
    neg_scores: list[list[float]],
    k: int,
    pos_ids: list[int],
) -> tuple[list[int], list[float]]:
    """
    Aggregate BM25 results from multiple queries
    """
    pos_scores = np.array(pos_scores)
    max_scalar = max(np.max(pos_scores), 1e-6)
    min_scalar = np.max(np.min(pos_scores, axis=1))
    pos_scores_avg = pos_scores.mean(axis=0)
    pos_scores_max = pos_scores.max(axis=0)
    # # check if combine avg and max with pos_ids
    # if if_combine_max_avg(pos_scores_avg, pos_scores_max, pos_ids, unique_objs):
    #     pos_scores = (pos_scores_avg + pos_scores_max) / 2
    #     print("Combine avg and max")
    # else:
    #     pos_scores = pos_scores_avg
    #     print("Only use avg")
    pos_scores = (pos_scores_avg + pos_scores_max) / 2
    # print(f"max_scalar: {max_scalar:.4f}, min_scalar: {min_scalar:.4f}")
    # score_distribution = np.percentile(pos_scores, list(range(0, 101, 5)))
    # print(f"{list(np.round(score_distribution, 4))}")
    if neg_scores is not None:
        neg_scores = np.mean(neg_scores, axis=0)
        best_param = choose_best_param(pos_scores, neg_scores, unique_objs, pos_ids)
        scores = pos_scores - neg_scores * best_param
    else:
        scores = pos_scores
    retrieved_ids, scores = zip(
        *sorted(zip(unique_objs, scores), key=lambda x: x[1], reverse=True)
    )
    retrieved_ids = retrieved_ids[:k]
    scores = scores[:k]
    scores = (np.array(scores) - min_scalar) / (max_scalar - min_scalar)
    return retrieved_ids, scores


def calculate_reformulate_impact(
    query_list: list[str],
    retrieved_info: RetrievedInfo,
    corpus: list[str],
    args,
    cur_iter_pos_objs: list[str],
    pos_ids: list[int] = [],
) -> float:
    """
    Calculate the impact of reformulating the query
    """
    # find the query that are from reformulated queries
    query_list = [q for q in query_list if q in retrieved_info.query_list]
    if len(query_list) == 0 or len(cur_iter_pos_objs) == 0:
        return 0
    print(f"[DEBUG] query from reformulated queries: {query_list}")
    reformulated_query_positions, other_query_positions = [], []
    bm25_pos_scores = retrieved_info.bm25_pos_scores
    hnsw_pos_scores = retrieved_info.hnsw_pos_scores
    bm25_pos_scores_new, hnsw_pos_scores_new = [], []
    for i, query in enumerate(retrieved_info.query_list):
        if query in query_list:
            reformulated_query_positions.append(i)
            bm25_pos_scores_new.append(bm25_pos_scores[i])
            hnsw_pos_scores_new.append(hnsw_pos_scores[i])
        else:
            other_query_positions.append(i)
    # get agg results without reformulated queries
    bm25_agg_ids, bm25_agg_scores = agg_results(
        retrieved_info.bm25_unique_ids,
        bm25_pos_scores_new,
        retrieved_info.bm25_neg_scores,
        args.k,
        pos_ids,
    )
    bm25_objs = [corpus[i] for i in bm25_agg_ids]
    hnsw_agg_ids, hnsw_agg_scores = agg_results(
        retrieved_info.hnsw_unique_ids,
        hnsw_pos_scores_new,
        retrieved_info.hnsw_neg_scores,
        args.k,
        pos_ids,
    )
    hnsw_objs = [corpus[i] for i in hnsw_agg_ids]
    impact_score = 0
    for obj in cur_iter_pos_objs:
        # obj is retrieved due to the reformulated queries
        if obj not in bm25_objs and obj not in hnsw_objs:
            impact_score += 1
    return impact_score
