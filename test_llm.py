import argparse
import pandas as pd
import json
import os
from parsing import parse_query
from copy import deepcopy
from utils import execute_sql
from load_data import load_data
import sqlite3
from llm_check import llm_check_batch, llm_check
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../TAG-Bench")
    parser.add_argument("--dataset", type=str, default="TAG")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--test_type", type=str, default="dev")
    parser.add_argument("--method", type=str, default="llm_check")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5)
    return parser.parse_args()


def save_results(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.dataset}.json")
    refined_output_path = os.path.join(args.output_dir, f"{args.dataset}_refined.json")
    # load data
    df, query_answer, query_template, path = load_data(args.dataset)
    if args.dataset == "paper":
        corpus = df["abstracts"].values.tolist()
        batch_size = 1024
        template = "The abstract of the paper is: {value}. Is this paper about {query}?"
        reformat_template = "According to the abstract, the paper is about {query}."
    elif args.dataset == "product":
        corpus = df["Product_Title"].values.tolist()
        batch_size = 512
        template = "The product title is: {value}. Is this product the same as or a type of '{query}'?"
        reformat_template = "According to the product title, the product is the same as or a type of '{query}'."
    else:
        cols = df.columns
        corpus = df[cols[0]].values.tolist()
        batch_size = 128
        template = "Is '{value}' the same as or a type of '{query}'?"
        reformat_template = (
            f"The {cols[0]}" + " that is the same as or a type of '{query}'."
        )
    db, table = args.dataset, args.dataset
    # solve query
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
        solved_queries = {result["query"] for result in results}
    else:
        results = []
        solved_queries = []
    refined_results = json.load(open(refined_output_path, "r"))
    refined_solved_queries = [result["query"] for result in refined_results]

    output_path = os.path.join(args.output_dir, f"{args.dataset}_v2.json")
    for d in results:
        query = d["query"]
        pred = d["pred"]
        if query not in refined_solved_queries:
            continue
        cond = reformat_template.format(query=query)
        filtered_pred = llm_check(query, corpus, template)
        d["pred"] = filtered_pred
        print(f"Query: {query}\nPred: {pred}\nFiltered Pred: {filtered_pred}")
        save_results(results, output_path)
    # cnt = 0
    # for query, answers in query_answer.items():
    #     if len(answers) == 0:
    #         continue
    #     cnt += 1
    #     if cnt > 50:
    #         break
    #     if query in solved_queries:
    #         print(f"Query {query} already solved, skipping.")
    #         continue
    #     start_time = time.time()
    #     try:
    #         pred = llm_check(query, corpus, template)
    #         result = {"query": query, "pred": pred, "answers": answers}
    #         result["time"] = time.time() - start_time
    #         results.append(result)
    #     except Exception as e:
    #         print(f"[ERROR] {e}")
    #         continue
    #     save_results(results, output_path)
    # # save results
    # save_results(results, output_path)
