import argparse
import pandas as pd
import json
import os
from load_data import load_data
import time
from reformulate import reformulate, score_query
from retrieve import retrieve_corpus
import numpy as np
from collections import defaultdict
from constants import ALPHA_DIFF, NEW_POS_RATIO
from llm_check import llm_check


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../TAG-Bench")
    parser.add_argument("--dataset", type=str, default="TAG")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--test_type", type=str, default="dev")
    parser.add_argument("--method", type=str, default="llm_check")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10)
    return parser.parse_args()


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


def combine_index(
    bm25_objs: list[str],
    hnsw_objs: list[str],
    bm25_scores: list[float],
    hnsw_scores: list[float],
    alpha: float,
) -> list[int]:
    """
    Combine the BM25 and HNSW indices
    """
    obj_scores = defaultdict(float)
    for i, bm25_score in enumerate(bm25_scores):
        obj_scores[bm25_objs[i]] = bm25_score * alpha
    for i, hnsw_score in enumerate(hnsw_scores):
        obj_scores[hnsw_objs[i]] += hnsw_score * (1 - alpha)
    obj_scores = sorted(obj_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_objs = [x[0] for x in obj_scores]
    return sorted_objs


def solve_query(
    query: str,
    attribute: str,
    corpus: list,
    args,
    path: str,
    reformat_template: str,
    llm_template: str,
    answers: list,
) -> dict:
    answers = [a.lower() for a in answers]
    # Step 1: load sample values, which are pre-selected
    table = args.dataset
    col_vals = json.load(open(f"{path}/{table}_sample_values.json"))
    # Step 2: Initialization
    reformulate_time, refine_time, retrieve_time = 0, 0, 0
    alpha = 0.5  # the weight of the BM25 index
    pos_objs = []  # the positive objects
    checked_obj_dict = {}
    query_scores = {query: 2}
    last_iter_pos_objs = []
    print(f"======Processing Query: [{query}]=======")
    for step in range(args.steps):
        cur_iter_pos_objs = []
        print(f"======Step {step}=======")
        # Reformulate the query
        start_time = time.time()
        if len(pos_objs) == 0:
            sample_values = col_vals[attribute]
        else:
            sample_values = pos_objs
        generated_query_list = reformulate(
            reformat_template.format(query=query), attribute, sample_values
        )
        print(f"Reformulated query: {generated_query_list}")
        reformulate_time += time.time() - start_time

        # score the query words in the query list
        start_time = time.time()
        new_query_objs = generated_query_list + last_iter_pos_objs
        query_scores_new = score_query(
            reformat_template.format(query=query), new_query_objs
        )
        query_scores.update(query_scores_new)
        refine_time += time.time() - start_time

        # retrieve the corpus based on the query list
        start_time = time.time()
        query_list = [q for q, s in query_scores.items() if s > 1]
        print(f"Query list: {query_list}")
        bm25_objs, bm25_scores, hnsw_objs, hnsw_scores = retrieve_corpus(
            query_scores,
            corpus,
            args,  # checked_obj_dict
        )
        retrieve_time += time.time() - start_time

        # iteratively examine the retrieved objs
        alpha_diff = 1
        while len(checked_obj_dict) < args.k:
            # rerank the objs if the alpha difference is greater than the threshold
            # check the objs from the start of the list
            if alpha_diff >= ALPHA_DIFF:
                sorted_objs = combine_index(
                    bm25_objs, hnsw_objs, bm25_scores, hnsw_scores, alpha
                )
                obj_idx = 0
            # check the next top_k objs
            objs_to_check = sorted_objs[obj_idx : obj_idx + args.top_k]
            new_pos_objs = llm_check(
                query, objs_to_check, llm_template, checked_obj_dict
            )
            cur_iter_pos_objs.extend(new_pos_objs)
            checked_obj_dict = update_checked_obj_dict(
                checked_obj_dict, objs_to_check, new_pos_objs
            )
            obj_idx += args.top_k
            # calculate the alpha based on the scores of the two indices
            bm25_score, hnsw_score = score_index(bm25_objs, hnsw_objs, checked_obj_dict)
            alpha_new = (
                bm25_score / (bm25_score + hnsw_score)
                if bm25_score + hnsw_score > 0
                else 0.5
            )
            alpha_diff = abs(alpha_new - alpha)
            alpha = alpha_new
            print(
                f"BM25 Score: {bm25_score:.4f}, HNSW Score: {hnsw_score:.4f}, "
                f"Alpha: {alpha:.4f}, Alpha Diff: {alpha_diff:.4f}"
            )
            if (
                len(cur_iter_pos_objs) >= 2
                and len(cur_iter_pos_objs) >= len(pos_objs) * NEW_POS_RATIO
            ):
                print(f"Step {step}: {len(cur_iter_pos_objs)} results found")
                break
            pos_num = 0
            for obj in objs_to_check:
                if checked_obj_dict.get(obj, 0) == 1:
                    pos_num += 1
            if pos_num == 0 and (len(pos_objs) > 0 or len(cur_iter_pos_objs) > 0):
                print(f"Step {step}: No results found for query: {query}")
                break
            else:
                print(f"Step {step}: new {pos_num} results found for query: {query}")
        pos_objs.extend(cur_iter_pos_objs)
        last_iter_pos_objs = list(cur_iter_pos_objs)
        if len(cur_iter_pos_objs) == 0:
            break
    return {
        "query": query,
        "pred": pos_objs,
        "retrieved": list(checked_obj_dict.keys()),
        "retrieved_num": len(checked_obj_dict),
        "reformulate_time": reformulate_time,
        "refine_time": refine_time,
        "retrieve_time": retrieve_time,
        "iteration_num": step + 1,
        "query_scores": query_scores,
    }


def save_results(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.dataset}.json")
    # load results
    if os.path.exists(output_path):
        results = json.load(open(output_path, "r"))
        results = [d for d in results if len(d["pred"]) > 0]
        processed_queries = [d["query"] for d in results]
    else:
        results = []
        processed_queries = []
    # load data
    df, query_answer, query_template, path = load_data(args.dataset)
    # TODO: rename the variables
    if args.dataset == "paper":
        batch_size = 1024
        attribute = "abstracts"
        reformat_template = "According to the abstract, the paper is about {query}."
        llm_template = (
            "The abstract of the paper is: {value}. Is this paper about {query}?"
        )
    elif args.dataset == "product":
        batch_size = 512
        attribute = "Product_Title"
        reformat_template = "According to the product title, the product is the same as or a type of '{query}'."
        llm_template = "Is '{value}' the same as or a type of '{query}'? Directly answer with 'Yes' or 'No'."
    else:
        cols = df.columns
        batch_size = 128
        attribute = cols[0]
        reformat_template = (
            f"The {attribute}" + " that is the same as or a type of '{query}'."
        )
        llm_template = "Is '{value}' the same as or a type of '{query}'? Directly answer with 'Yes' or 'No'."
    corpus = df[attribute].values.tolist()

    # solve query
    print(
        f"Total queries: {len(query_answer)}, processed queries: {len(processed_queries)}"
    )
    cnt = 0
    for query, answers in query_answer.items():
        if query in processed_queries:
            continue
        if len(answers) == 0:
            continue
        start_time = time.time()
        result = solve_query(
            query,
            attribute,
            corpus,
            args,
            path,
            reformat_template,
            llm_template,
            answers,
        )
        result["answers"] = answers
        result["time"] = time.time() - start_time
        results.append(result)
        save_results(results, output_path)
    # save results
    save_results(results, output_path)
