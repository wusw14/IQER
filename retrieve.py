from sentence_transformers import SentenceTransformer
import numpy as np
from index import BM25Index, HNSWIndex
from pydantic import BaseModel


class RetrievedInfo(BaseModel):
    bm25_objs: list[str]
    bm25_scores: list[float]
    hnsw_objs: list[str]
    hnsw_scores: list[float]


def retrieve_corpus(
    query_scores,
    corpus: list[str],
    args: dict,
    checked_results: dict[str, int] = {},
) -> RetrievedInfo:
    """
    Retrieve corpus from BM25 and HNSW index
    """
    bm25_index = BM25Index(corpus, args.dataset)
    hnsw_index = HNSWIndex(corpus, args.dataset)
    if type(query_scores) == list:
        query_list = list(query_scores)
        weights = np.array([1] * len(query_list))
    elif type(query_scores) == dict:
        query_list_w1, query_list_w2 = [], []
        for q, s in query_scores.items():
            if s == 2:
                query_list_w2.append(q)
            elif s == 1:
                query_list_w1.append(q)
        if len(query_list_w1) > len(query_list_w2):
            query_list_w1 = np.random.choice(query_list_w1, len(query_list_w2))
            query_list_w1 = query_list_w1.tolist()
        query_list = query_list_w1 + query_list_w2
        weights = np.array([query_scores[q] / 2.0 for q in query_list])
    else:
        raise ValueError(f"Invalid query_scores type: {type(query_scores)}")
    weights = weights.reshape(-1, 1)

    if len(checked_results) > 0:
        pos_queries = [q for q, v in checked_results.items() if v > 0]
        pos_ids = [corpus.index(q) for q in pos_queries]
        neg_queries = [q for q, v in checked_results.items() if v == 0]
        if len(neg_queries) > 5:
            neg_queries = np.random.choice(neg_queries, 5)
    else:
        neg_queries = []
        pos_ids = []

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
        pos_ids,
    )
    bm25_objs = [corpus[i] for i in bm25_results]
    # normalize bm25 scores
    bm25_scores = np.array(bm25_scores)
    # bm25_scores = (bm25_scores - bm25_scores[-1]) / (
    #     bm25_scores[0] - bm25_scores[-1] + 1e-6
    # )
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
        query_embs,
        unique_results,
        results_embs,
        neg_query_emb,
        weights,
        args.k,
        pos_ids,
    )
    hnsw_objs = [corpus[i] for i in hnsw_results]
    # normalize hnsw scores
    hnsw_scores = np.array(hnsw_scores)
    # hnsw_scores = (hnsw_scores - hnsw_scores[-1]) / (
    #     hnsw_scores[0] - hnsw_scores[-1] + 1e-6
    # )
    retrieved_info = RetrievedInfo(
        bm25_objs=bm25_objs,
        bm25_scores=bm25_scores,
        hnsw_objs=hnsw_objs,
        hnsw_scores=hnsw_scores,
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


def agg_bm25_results(
    bm25_results: list[list[int]],
    pos_bm25_scores: list[list[float]],
    neg_bm25_scores,
    weights: np.ndarray,
    k: int,
    pos_ids: list[int],
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
    max_scalar = max(np.max(pos_bm25_scores), 1e-6)
    min_scalar = np.max(np.min(pos_bm25_scores, axis=1))
    pos_scores_avg = pos_bm25_scores.sum(axis=0) / np.sum(weights)
    pos_scores_max = np.max(pos_bm25_scores, axis=0)
    # check if combine avg and max with pos_ids
    if if_combine_max_avg(pos_scores_avg, pos_scores_max, pos_ids, retrieved_ids):
        pos_scores = (pos_scores_avg + pos_scores_max) / 2
        print("Combine avg and max")
    else:
        pos_scores = pos_scores_avg
        print("Only use avg")
    print(f"BM25 max_scalar: {max_scalar:.4f}, min_scalar: {min_scalar:.4f}")
    score_distribution = np.percentile(pos_scores, list(range(0, 101, 5)))
    print(f"{list(np.round(score_distribution, 4))}")
    if neg_bm25_scores is not None:
        neg_bm25_scores = np.array(neg_bm25_scores)
        neg_bm25_scores = neg_bm25_scores[:, retrieved_ids]
        neg_scores = np.mean(neg_bm25_scores, axis=0)
        best_param = choose_best_param(pos_scores, neg_scores, retrieved_ids, pos_ids)
        scores = pos_scores - neg_scores * best_param
    else:
        scores = pos_scores
    retrieved_ids, scores = zip(
        *sorted(zip(retrieved_ids, scores), key=lambda x: x[1], reverse=True)
    )
    retrieved_ids = retrieved_ids[:k]
    scores = scores[:k]
    scores = (np.array(scores) - min_scalar) / (max_scalar - min_scalar)
    return retrieved_ids, scores


def agg_hnsw_results(
    query_embs: np.ndarray,
    retrieved_ids: list[int],
    results_embs: np.ndarray,
    neg_query_emb,
    weights: np.ndarray,
    k: int,
    pos_ids: list[int],
) -> tuple[list[int], list[float]]:
    """
    Aggregate HNSW results from multiple queries
    """
    # normalize the query embeddings
    query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    # calculate the similarity between the query and the results
    pos_scores = np.dot(query_embs, results_embs.T)
    pos_scores = pos_scores * weights
    max_scalar = max(np.max(pos_scores), 1e-6)
    min_scalar = np.max(np.min(pos_scores, axis=1))
    print(f"HNSW max_scalar: {max_scalar:.4f}, min_scalar: {min_scalar:.4f}")
    pos_scores_avg = pos_scores.sum(axis=0) / np.sum(weights)
    pos_scores_max = np.max(pos_scores, axis=0)
    # check if combine avg and max with pos_ids
    if if_combine_max_avg(pos_scores_avg, pos_scores_max, pos_ids, retrieved_ids):
        pos_scores = (pos_scores_avg + pos_scores_max) / 2
        print("Combine avg and max")
    else:
        pos_scores = pos_scores_avg
        print("Only use avg")
    score_distribution = np.percentile(pos_scores, list(range(0, 101, 5)))
    print(f"{list(np.round(score_distribution, 4))}")
    if neg_query_emb is not None:
        neg_scores = np.dot(neg_query_emb, results_embs.T)
        best_param = choose_best_param(pos_scores, neg_scores, retrieved_ids, pos_ids)
        scores = pos_scores - neg_scores * best_param
    else:
        scores = pos_scores
    retrieved_ids, scores = zip(
        *sorted(zip(retrieved_ids, scores), key=lambda x: x[1], reverse=True)
    )
    retrieved_ids = retrieved_ids[:k]
    scores = scores[:k]
    scores = (np.array(scores) - min_scalar) / (max_scalar - min_scalar)
    return retrieved_ids, scores


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
