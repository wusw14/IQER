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
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--test_type", type=str, default="dev")
    parser.add_argument("--method", type=str, default="llm_check")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10)
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
    # query_list, checked_results, if_reformuation = warm_up(
    #     query, corpus, args, llm_template, reformat_template, attribute, col_vals
    # )
    query_list = [query]
    checked_results = {}
    if_reformuation = False
    warm_up_time = time.time() - start_time
    # Step 3: iteratively examine the results retrieved by BM25 and HNSW
    pos_objs = [k for k, v in checked_results.items() if v == 1]
    print(f"Pos objs: {pos_objs}")
    print(f"Checked results: {checked_results}")
    for step in range(args.steps):
        print(f"======Step {step}=======")
        # retrieve the corpus based on the query list
        query_list = list(set(query_list) | set(pos_objs))
        print(f"Query list: {query_list}")
        bm25_results, hnsw_results = retrieve_corpus(query_list, corpus, args)
        bm25_idx, hnsw_idx = 0, 0
        bm25_score, hnsw_score = score_index(
            bm25_results, hnsw_results, checked_results
        )
        while len(checked_results) < args.k:
            bm25_objs = bm25_results[bm25_idx : bm25_idx + args.top_k]
            hnsw_objs = hnsw_results[hnsw_idx : hnsw_idx + args.top_k]
            objs_to_check = []
            if bm25_score == hnsw_score:
                bm25_idx += args.top_k
                hnsw_idx += args.top_k
                objs_to_check = list(set(bm25_objs + hnsw_objs))
                objs_type = "both"
            elif bm25_score > hnsw_score:
                objs_to_check = bm25_objs
                bm25_idx += args.top_k
                objs_type = "bm25"
            else:
                objs_to_check = hnsw_objs
                hnsw_idx += args.top_k
                objs_type = "hnsw"
            filtered_objs = llm_check(
                query, objs_to_check, llm_template, checked_results
            )
            pos_objs.extend(filtered_objs)
            checked_results = update_checked_results(
                checked_results, objs_to_check, filtered_objs
            )
            if objs_type == "both":
                bm25_score, hnsw_score = score_index(
                    bm25_objs, hnsw_objs, checked_results
                )
            elif objs_type == "bm25":
                bm25_score, _ = score_index(bm25_objs, hnsw_objs, checked_results)
            else:
                _, hnsw_score = score_index(bm25_objs, hnsw_objs, checked_results)
            print(f"Obj: {objs_type}, BM25: {bm25_score}, HNSW: {hnsw_score}")
            # if len(filtered_objs) >= 2 and len(filtered_objs) >= len(pos_objs) * 0.25:
            #     print(f"Step {step}: {len(filtered_objs)} results found")
            #     break
            if bm25_score == 0 and hnsw_score == 0:
                print(f"No results found for query: {query}")
                break
        if len(filtered_objs) == 0:
            break
        # evaluate the two sets of index
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
        "warm_up_time": warm_up_time,
        "iteration_num": step + 1,
        "if_reformuation": if_reformuation,
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
        processed_queries = [d["query"] for d in results]
    else:
        results = []
        processed_queries = []
    # load data
    df, query_answer, query_template, path = load_data(args.dataset)
    if args.dataset == "paper":
        corpus = df["abstracts"].values.tolist()
        batch_size = 1024
        attribute = "abstracts"
        reformat_template = "According to the abstract, the paper is about {query}."
        llm_template = (
            "The abstract of the paper is: {value}. Is this paper about {query}?"
        )
    elif args.dataset == "product":
        corpus = df["Product_Title"].values.tolist()
        batch_size = 512
        attribute = "Product_Title"
        reformat_template = "According to the product title, the product is the same as or a type of '{query}'."
        # llm_template = "The product title is: {value}. Is this product the same as or a type of '{query}'?"
        llm_template = "Is '{value}' the same as or a type of '{query}'? Directly answer with 'Yes' or 'No'."
    else:
        cols = df.columns
        corpus = df[cols[0]].values.tolist()
        batch_size = 128
        attribute = cols[0]
        reformat_template = (
            # f"The {attribute}" + " that is the same as or a type of '{query}'."
            f"The {attribute}"
            + " is '{query}'."
        )
        # llm_template = "Is '{value}' the same as '{query}' or a type of '{query}'?"
        llm_template = "Is '{value}' the same as or a type of '{query}'? Directly answer with 'Yes' or 'No'."

    # if not os.path.exists(f"data/db/{args.dataset}.db"):
    #     os.makedirs(f"data/db", exist_ok=True)
    #     conn = sqlite3.connect(f"data/db/{args.dataset}.db")
    #     df.to_sql(args.dataset, conn, index=False)
    # db, table = args.dataset, args.dataset
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
        try:
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
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
        save_results(results, output_path)
    # save results
    save_results(results, output_path)
