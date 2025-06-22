import argparse
import pandas as pd
import json
import os
from parsing import parse_query
from copy import deepcopy
from utils import execute_sql
from load_data import load_data
import sqlite3
from llm_check import llm_check, llm_check_batch
import time
from reformulate import reformulate, refine_query
from retrieve import retrieve_corpus
import numpy as np
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../TAG-Bench")
    parser.add_argument("--dataset", type=str, default="TAG")
    parser.add_argument("--test_type", type=str, default="dev")
    parser.add_argument("--method", type=str, default="llm_check")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument(
        "--index_combine_method",
        type=str,
        choices=["weighted", "merge"],
        default="weighted",
    )
    parser.add_argument("--exp_name", type=str, default="debug")
    return parser.parse_args()


def combine_index(bm25_results, hnsw_results, args):
    item_scores = defaultdict(float)
    for bm_loc, bm_result in enumerate(bm25_results):
        item_scores[bm_result] = 1 / np.log2(bm_loc / args.alpha + 2)
    for hnsw_loc, hnsw_result in enumerate(hnsw_results):
        item_scores[hnsw_result] += 1 / np.log2(hnsw_loc / args.alpha + 2)
    results = sorted(item_scores, key=lambda x: item_scores[x], reverse=True)
    results = results[: args.top_k]
    bm25_results_tobe_checked, hnsw_results_tobe_checked = [], []
    for item in results:
        if item in bm25_results:
            bm25_results_tobe_checked.append(item)
        if item in hnsw_results:
            hnsw_results_tobe_checked.append(item)
    return results, bm25_results_tobe_checked, hnsw_results_tobe_checked


def warm_up(
    query: str,
    corpus: list,
    args,
    llm_template: str,
    reformat_template: str,
    attribute: str,
    col_vals: dict,
) -> list:
    print("======Warm-up=======")
    query_list = [query]
    print(f"Query: {query_list}")
    bm25_results, hnsw_results = retrieve_corpus(query_list, corpus, args)
    checked_results = {}
    # score the intersection of the results
    results, bm25_results, hnsw_results = combine_index(
        bm25_results, hnsw_results, args
    )
    print(f"Candidates: {results}")
    for result in results:
        checked_results[result] = 0
    # let LLM check the results
    filtered_results = llm_check(query, results, llm_template)
    print(f"Filtered: {filtered_results}")
    for result in filtered_results:
        checked_results[result] = 1
    # TODO: test the impact of query reformulation
    query_list_generated = reformulate(
        reformat_template.format(query=query), attribute, col_vals[attribute]
    )
    if "unknown" in query_list_generated:
        query_list_generated = []
    query_list = list(set(query_list_generated + filtered_results))
    query_list = refine_query(reformat_template.format(query=query), query_list)
    if query not in query_list and query.lower() not in query_list:
        query_list.append(query)
    reformuation = True
    # if len(filtered_results) == 0 or (
    #     len(filtered_results) == 1 and (filtered_results[0]).lower() == query.lower()
    # ):
    #     reformuation = True
    #     # reformulate the query
    #     query_list = reformulate(
    #         reformat_template.format(query=query), attribute, col_vals[attribute]
    #     )
    #     if "unknown" in query_list:
    #         query_list = [query]
    #     else:
    #         for q in query_list:
    #             checked_results[q] = 0
    #         # refine the query list
    #         query_list = refine_query(reformat_template.format(query=query), query_list)
    #         for q in query_list:
    #             checked_results[q] = 1
    # else:
    #     reformuation = False
    #     query_list = list(set(filtered_results + query_list))
    return query_list, checked_results, reformuation


def score_index(bm25_results, hnsw_results, checked_results):
    if len(checked_results) == 0:
        return 0, 0
    # calculate NDCG for bm25 and hnsw
    bm25_ndcg, hnsw_ndcg = 0, 0
    for k, v in checked_results.items():
        if v == 1:
            if k in bm25_results:
                bm25_ndcg += 1 / np.log2(bm25_results.index(k) + 2)
            if k in hnsw_results:
                hnsw_ndcg += 1 / np.log2(hnsw_results.index(k) + 2)
    return bm25_ndcg, hnsw_ndcg


def update_checked_results(checked_results, objs_to_check, filtered_objs):
    for obj in objs_to_check:
        if obj in checked_results:
            continue
        if obj in filtered_objs:
            checked_results[obj] = 1
        else:
            checked_results[obj] = 0
    return checked_results


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
    # Step 2: warm-up the querying process by enriching the query
    start_time = time.time()
    query_list = [query]
    checked_results = {}
    if_reformuation = False
    warm_up_time = time.time() - start_time
    print(f"Query list: {query_list}")
    retrieved_info = retrieve_corpus(query_list, corpus, args)
    bm25_results = retrieved_info.bm25_objs
    bm25_scores = retrieved_info.bm25_scores
    hnsw_results = retrieved_info.hnsw_objs
    hnsw_scores = retrieved_info.hnsw_scores
    # aggregate scores from two indices
    obj_scores = defaultdict(float)
    for bm_obj, bm_score in zip(bm25_results, bm25_scores):
        obj_scores[bm_obj] += bm_score * args.alpha
    for hnsw_obj, hnsw_score in zip(hnsw_results, hnsw_scores):
        obj_scores[hnsw_obj] += hnsw_score * (1 - args.alpha)
    # sort the objects by scores
    sorted_obj_score = sorted(obj_scores.items(), key=lambda x: x[1], reverse=True)
    # get top k objects
    sorted_objs = [obj for obj, _ in sorted_obj_score]
    idx = 0
    while len(checked_results) < args.k:
        objs_to_check = sorted_objs[idx : idx + args.top_k]
        filtered_objs = llm_check(query, objs_to_check, llm_template, checked_results)
        checked_results = update_checked_results(
            checked_results, objs_to_check, filtered_objs
        )
        idx += args.top_k
    checked_objs = list(checked_results.keys())
    results = []
    for k, v in checked_results.items():
        if v == 1:
            results.append(k)
    return {
        "query": query,
        "pred": results,
        "retrieved": checked_objs,
        "retrieved_num": len(checked_objs),
    }


def save_results(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    args.output_dir = f"results/{args.exp_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{args.alpha}.json")
    # load results
    if os.path.exists(output_path) and args.exp_name != "debug":
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
            f"According to the {attribute} name, the {attribute}"
            + " is the same as or a type of '{query}'."
        )
        llm_template = "Is '{value}' the same as or a type of '{query}'? Directly answer with 'Yes' or 'No'."
    corpus = df[attribute].values.tolist()

    args.llm_template = llm_template

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
        cnt += 1
        if args.exp_name == "debug" and cnt >= 5:
            break
    # save results
    save_results(results, output_path)
