from collections import defaultdict

import numpy as np
from constants import ALPHA, NEW_POS_RATIO
from llm_check import llm_check
from retrieve import RetrievedInfo


def iterative_check_retrieved_objs(
    query: str,
    retrieved_info: RetrievedInfo,
    args,
    checked_obj_dict: dict[str, int],
    step: int,
):
    if args.index_combine_method == "weighted":
        cur_iter_pos_objs, checked_obj_dict = weighted_combine_check_objs(
            query, retrieved_info, args, checked_obj_dict, step
        )
    elif args.index_combine_method == "merge":
        cur_iter_pos_objs, checked_obj_dict = merge_combine_check_objs(
            query, retrieved_info, args, checked_obj_dict, step
        )
    else:
        raise ValueError(f"Invalid method: {args.method}")
    return cur_iter_pos_objs, checked_obj_dict


def update_checked_obj_dict(
    checked_obj_dict: dict[str, int],
    objs_to_check: list[str],
    filtered_objs: list[str],
) -> dict[str, int]:
    """
    Update the checked objects dictionary
    """
    for obj in objs_to_check:
        if obj in checked_obj_dict:
            continue
        if obj in filtered_objs:
            checked_obj_dict[obj] = 1
        else:
            checked_obj_dict[obj] = 0
    return checked_obj_dict


def weighted_combine_check_objs(
    query: str,
    retrieved_info: RetrievedInfo,
    args,
    checked_obj_dict: dict[str, int],
    step: int,
):
    sorted_objs = combine_index(retrieved_info, ALPHA)
    sorted_objs = [s for s in sorted_objs if s not in checked_obj_dict]
    obj_idx = 0
    early_stop = 0
    cur_iter_pos_objs = []
    past_pos_num = sum(checked_obj_dict.values())
    while len(checked_obj_dict) < args.k:
        # check the next top_k objs
        objs_to_check = sorted_objs[obj_idx : obj_idx + args.top_k]
        objs_to_check = objs_to_check[: args.k - len(checked_obj_dict)]
        new_pos_objs = llm_check(
            query, objs_to_check, args.llm_template, checked_obj_dict
        )
        cur_iter_pos_objs.extend(new_pos_objs)
        checked_obj_dict = update_checked_obj_dict(
            checked_obj_dict, objs_to_check, new_pos_objs
        )
        obj_idx += args.top_k
        print(f"Checked objects: {objs_to_check}")
        print(f"New positive objects: {new_pos_objs}")
        if (
            len(cur_iter_pos_objs) >= 2
            and len(cur_iter_pos_objs) >= past_pos_num * NEW_POS_RATIO
            and step < args.steps - 1
        ):
            print(f"Step {step}: {len(cur_iter_pos_objs)} results found")
            break
        # pos_num = 0
        # for obj in objs_to_check:
        #     if checked_obj_dict.get(obj, 0) == 1:
        #         pos_num += 1
        # if pos_num == 0:
        #     if len(cur_iter_pos_objs) > 0:
        #         print(f"Step {step}: No more results found for query: {query}")
        #         break
        #     else:
        #         early_stop += 1
        #         if early_stop >= 2 and past_pos_num > 0:
        #             print(f"Step {step}: Early stop for query: {query}")
        #             break
        # else:
        #     early_stop = 0
        #     print(f"Step {step}: new {pos_num} results found for query: {query}")
    return cur_iter_pos_objs, checked_obj_dict


def merge_combine_check_objs(
    query: str,
    retrieved_info: RetrievedInfo,
    args,
    checked_obj_dict: dict[str, int],
    step: int,
):
    bm25_objs = retrieved_info.bm25_objs
    hnsw_objs = retrieved_info.hnsw_objs
    bm25_idx, hnsw_idx = 0, 0
    cur_iter_pos_objs = []
    past_pos_num = sum(checked_obj_dict.values())
    bm25_objs_tobe_checked = bm25_objs[: args.top_k]
    hnsw_objs_tobe_checked = hnsw_objs[: args.top_k]
    bm25_score, hnsw_score = score_index(
        bm25_objs_tobe_checked, hnsw_objs_tobe_checked, checked_obj_dict
    )
    print(f"BM25 score: {bm25_score:.4f}, HNSW score: {hnsw_score:.4f}")
    early_stop = 0
    while len(checked_obj_dict) < args.k:
        if bm25_score > hnsw_score:
            objs_to_check = bm25_objs[bm25_idx : bm25_idx + args.top_k]
            bm25_idx += args.top_k
            bm25_objs_tobe_checked = objs_to_check
        elif bm25_score < hnsw_score:
            objs_to_check = hnsw_objs[hnsw_idx : hnsw_idx + args.top_k]
            hnsw_idx += args.top_k
            hnsw_objs_tobe_checked = objs_to_check
        else:
            objs_to_check = bm25_objs[bm25_idx : bm25_idx + args.top_k]
            objs_to_check.extend(hnsw_objs[hnsw_idx : hnsw_idx + args.top_k])
            bm25_idx += args.top_k
            hnsw_idx += args.top_k
            bm25_objs_tobe_checked = objs_to_check[: args.top_k]
            hnsw_objs_tobe_checked = objs_to_check[args.top_k :]
        objs_to_check = list(set(objs_to_check))
        objs_to_check = [o for o in objs_to_check if o not in checked_obj_dict]
        objs_to_check = objs_to_check[: args.k - len(checked_obj_dict)]
        new_pos_objs = llm_check(
            query, objs_to_check, args.llm_template, checked_obj_dict
        )
        cur_iter_pos_objs.extend(new_pos_objs)
        checked_obj_dict = update_checked_obj_dict(
            checked_obj_dict, objs_to_check, new_pos_objs
        )
        bm25_score, hnsw_score = score_index(
            bm25_objs_tobe_checked, hnsw_objs_tobe_checked, checked_obj_dict
        )
        print(f"BM25 score: {bm25_score:.4f}, HNSW score: {hnsw_score:.4f}")
        if (
            len(cur_iter_pos_objs) >= 2
            and len(cur_iter_pos_objs) >= past_pos_num * NEW_POS_RATIO
            and step < args.steps - 1
        ):
            print(f"Step {step}: {len(cur_iter_pos_objs)} results found")
            break
        # elif bm25_score == 0 and hnsw_score == 0:
        #     if len(cur_iter_pos_objs) > 0:
        #         print(f"Step {step}: No more results found for query: {query}")
        #         break
        #     else:
        #         early_stop += 1
        #         if early_stop >= 2 and past_pos_num > 0:
        #             print(f"Step {step}: Early stop for query: {query}")
        #             break
        # else:
        #     early_stop = 0
    return cur_iter_pos_objs, checked_obj_dict


def combine_index(
    retrieved_info: RetrievedInfo,
    alpha: float,
) -> list[int]:
    """
    Combine the BM25 and HNSW indices
    """
    bm25_objs = retrieved_info.bm25_objs
    hnsw_objs = retrieved_info.hnsw_objs
    bm25_scores = retrieved_info.bm25_scores
    hnsw_scores = retrieved_info.hnsw_scores
    obj_scores = defaultdict(float)
    for i, bm25_score in enumerate(bm25_scores):
        obj_scores[bm25_objs[i]] = bm25_score * alpha
    for i, hnsw_score in enumerate(hnsw_scores):
        obj_scores[hnsw_objs[i]] += hnsw_score * (1 - alpha)
    obj_scores = sorted(obj_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_objs = [x[0] for x in obj_scores]
    return sorted_objs


def score_index(
    bm25_objs: list[int], hnsw_objs: list[int], checked_obj_dict: dict[str, int]
) -> tuple[float, float]:
    """
    Score the index based on the checked objects
    """
    if len(checked_obj_dict) == 0:
        return 0, 0
    # calculate NDCG for bm25 and hnsw
    bm25_ndcg, hnsw_ndcg = 0, 0
    for k, v in checked_obj_dict.items():
        if v == 1:
            if k in bm25_objs:
                bm25_ndcg += 1 / np.log2(bm25_objs.index(k) + 2)
            if k in hnsw_objs:
                hnsw_ndcg += 1 / np.log2(hnsw_objs.index(k) + 2)
    return bm25_ndcg, hnsw_ndcg
