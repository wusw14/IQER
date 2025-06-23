import argparse
import pandas as pd
import json
import os
from load_data import load_data
import time
from reformulate import reformulate, score_query, if_reformulate
from retrieve import retrieve_corpus, calculate_reformulate_impact
from iterative_check import iterative_check_retrieved_objs
from query import Query
import numpy as np
from index import BM25Index, HNSWIndex


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


def get_sample_values(checked_obj_dict: dict[str, int]) -> list[str]:
    """
    Get the sample values based on the checked objects
    """
    pos_objs = [k for k, v in checked_obj_dict.items() if v == 1]
    neg_objs = [k for k, v in checked_obj_dict.items() if v == 0]
    if len(neg_objs) > 5:
        neg_objs = np.random.choice(neg_objs, 5, replace=False)
    samples = list(pos_objs) + list(neg_objs)
    return samples


def solve_query(
    query: Query,
    attribute: str,
    corpus: list,
    args,
    answers: list,
) -> dict:
    answers = [a.lower() for a in answers]
    # Step 1: Initialization
    reformulate_time, refine_time, retrieve_time, check_time = 0, 0, 0, 0
    checked_obj_dict = {}
    reformulate_impact = 0
    print(f"\n\n\n======Processing Query: [{query.org_query}]=======")
    # Step 2: Iteratively refine the query and retrieve the cell values
    query_list = [query.org_query]
    neg_list = []
    bm25_index = BM25Index(corpus, args.dataset)
    hnsw_index = HNSWIndex(corpus, args.dataset)
    for step in range(args.steps):
        cur_generated_query_list = []

        print(f"\n======Step {step}=======")
        # Reformulate the query
        start_time = time.time()
        if reformulate_impact > 0 or step == 1:
            sample_values = get_sample_values(checked_obj_dict)
            cur_generated_query_list = reformulate(
                query.query_condition,
                attribute,
                sample_values,
                query.queries_from_generated,
            )
        query.update_queries_from_generated(cur_generated_query_list)
        print(f"Reformulated query: {cur_generated_query_list}")
        reformulate_time += time.time() - start_time
        print(f"Time for reformulating: {time.time() - start_time:.4f}s")
        # score and select the diversified query words from the query list
        start_time = time.time()
        new_query_objs = query.new_queries_from_generated + query.new_queries_from_table
        if len(new_query_objs) > 0:
            query_scores_new = score_query(query.query_condition, new_query_objs)
            query.update_query_scores(query_scores_new)
            # select the diversified query words from the query list
            query_list = query.select_diversified_query_words(hnsw_index.emb_model)
            print(
                f"Step {step} Diversified query list ({len(query_list)}): {query_list}"
            )
            # neg_list = query.get_neg_list(checked_obj_dict)
            # print(f"Step {step} Neg list ({len(neg_list)}): {neg_list}")
            neg_list = []
        else:
            print(f"Step {step} Use the last step's query list")
        refine_time += time.time() - start_time
        print(f"Time for refining: {time.time() - start_time:.4f}s")

        # retrieve the corpus based on the query list
        if query.last_query_list != query_list:
            start_time = time.time()
            retrieved_info = retrieve_corpus(
                query_list,
                corpus,
                args,
                bm25_index,
                hnsw_index,
                neg_list,
                query.queries_from_table,
            )
            retrieve_time += time.time() - start_time
            query.last_query_list = query_list
            print(f"Time for retrieving: {time.time() - start_time:.4f}s")
        # iteratively examine the retrieved objs
        start_time = time.time()
        cur_iter_pos_objs, checked_obj_dict = iterative_check_retrieved_objs(
            query,
            retrieved_info,
            args,
            checked_obj_dict,
            step,
        )
        check_time += time.time() - start_time
        query.update_queries_from_table(cur_iter_pos_objs, corpus)
        if len(cur_iter_pos_objs) == 0 and step > 0 and len(checked_obj_dict) >= args.k:
            break
        reformulate_impact = calculate_reformulate_impact(
            query.new_queries_from_generated,
            retrieved_info,
            corpus,
            args,
            cur_iter_pos_objs,
            query.pos_ids,
        )
        print(f"Reformulate impact: {reformulate_impact}")

    if len(query.new_queries_from_table) > 0:
        query_scores_new = score_query(
            query.query_condition, query.new_queries_from_table
        )
        query.update_query_scores(query_scores_new)

    return {
        "query": query.org_query,
        "pred": query.queries_from_table,
        "retrieved": list(checked_obj_dict.keys()),
        "retrieved_num": len(checked_obj_dict),
        "reformulate_time": reformulate_time,
        "refine_time": refine_time,
        "retrieve_time": retrieve_time,
        "check_time": check_time,
        "iteration_num": step + 1,
        "query_scores": query.query_scores,
    }


def save_results(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    args.output_dir = f"results/{args.exp_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, f"{args.dataset}_{args.index_combine_method}.json"
    )
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
        query = Query(query, reformat_template)
        result = solve_query(query, attribute, corpus, args, answers)
        result["answers"] = answers
        result["time"] = time.time() - start_time
        results.append(result)
        save_results(results, output_path)
        cnt += 1
        if args.exp_name == "debug" and cnt >= 5:
            break
    # save results
    save_results(results, output_path)
