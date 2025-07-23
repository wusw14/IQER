import argparse
import json
import os
from load_data import load_data
import time
from reformulate import reformulate, score_query
from retrieve import retrieve_corpus
from iterative_check import llm_check_retrieved_objs
from query import Query
import numpy as np
from index import BM25Index, HNSWIndex
from rerank import rerank_retrieved_objs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../TAG-Bench")
    parser.add_argument("--dataset", type=str, default="TAG")
    parser.add_argument("--test_type", type=str, default="dev")
    parser.add_argument("--method", type=str, default="llm_check")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--rethink", action="store_true")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument(
        "--index_combine_method",
        type=str,
        choices=["weighted", "merge"],
        default="weighted",
    )
    parser.add_argument("--exp_name", type=str, default="debug")
    return parser.parse_args()


def get_sample_values(
    checked_obj_dict: dict[str, int], args, attribute: str
) -> list[str]:
    """
    Get the sample values based on the checked objects
    """
    if len(checked_obj_dict) == 0:
        temp_data = json.load(open(f"{args.path}/{args.dataset}_sample_values.json"))
        samples = temp_data[attribute]
    else:
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
    bm25_index: BM25Index,
    hnsw_index: HNSWIndex,
) -> dict:
    answers = [a.lower() for a in answers]
    # Step 1: Initialization
    reformulate_time, refine_time, retrieve_time, check_time = 0, 0, 0, 0
    rerank_time = 0
    print(f"\n\n\n======Processing Query: [{query.org_query}]=======")

    # Step 3: Iteratively refine the query and retrieve the cell values
    early_stop = 0
    step = 0
    # step 3.1: score and select the diversified query words from the query list
    bm25_queries = [query.org_query]
    query_list = [query.org_query]

    # step 3.2: retrieve the corpus based on the query list
    start_time = time.time()
    retrieved_info = retrieve_corpus(
        bm25_queries,
        query_list,
        corpus,
        args,
        bm25_index,
        hnsw_index,
        query.pos_ids,
        [],  # query.neg_ids,
    )
    retrieve_time += time.time() - start_time
    print(f"Time for retrieving: {time.time() - start_time:.4f}s")

    # step 3.3: rerank the retrieved objs and select the objs to be checked
    start_time = time.time()
    bm25_objs = retrieved_info.bm25_objs
    hnsw_objs = retrieved_info.hnsw_objs
    bm25_agg_pos_scores = retrieved_info.bm25_agg_pos_scores
    hnsw_agg_pos_scores = retrieved_info.hnsw_agg_pos_scores
    bm25_obj_score_dict = {
        obj: score for obj, score in zip(bm25_objs, bm25_agg_pos_scores)
    }
    hnsw_obj_score_dict = {
        obj: score for obj, score in zip(hnsw_objs, hnsw_agg_pos_scores)
    }
    obj_score = {}
    for obj in set(bm25_objs) | set(hnsw_objs):
        score1 = bm25_obj_score_dict.get(obj, 0)
        score2 = hnsw_obj_score_dict.get(obj, 0)
        obj_score[obj] = score1 * args.alpha + score2 * (1 - args.alpha)
    sorted_obj_score = sorted(obj_score.items(), key=lambda x: x[1], reverse=True)
    obj_to_check = [obj for obj, _ in sorted_obj_score[: args.budget]]

    rerank_time += time.time() - start_time
    print(f"Time for reranking: {time.time() - start_time:.4f}s")
    print(f"obj_to_check: {obj_to_check}")

    # step 3.4: llm check the retrieved objs
    start_time = time.time()
    obj_scores, query_scores = llm_check_retrieved_objs(query, obj_to_check, args)
    query.update_query_scores(query_scores)
    query_from_table = [q for q, s in obj_scores.items() if s > 0]
    query.update_queries_from_table(query_from_table)
    query.update_obj_scores(obj_scores, corpus)
    check_time += time.time() - start_time
    print(f"check {len(obj_scores)} objs")
    print(f"query_scores: {query_scores}")
    print(f"Time for checking: {time.time() - start_time:.4f}s")

    return {
        "query": query.org_query,
        "pred": query.queries_from_table,
        "retrieved": list(query.obj_scores.keys()),
        "retrieved_num": len(query.obj_scores),
        "query_scores": query.query_scores,
        "reformulate_time": reformulate_time,
        "refine_time": refine_time,
        "retrieve_time": retrieve_time,
        "rerank_time": rerank_time,
        "check_time": check_time,
        "iteration_num": step + 1,
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
    args.path = path
    # TODO: rename the variables
    if args.dataset == "product":
        batch_size = 512
        attribute = "Product_Title"
        # reformat_template = "According to the product title, the product is the same as or a type of '{query}'."
    else:
        cols = df.columns
        batch_size = 128
        attribute = cols[0]
        # reformat_template = (
        #     f"According to the {attribute} name, the {attribute}"
        #     + " is the same as or a type of '{query}'."
        # )
        # reformat_template = "The value is the same as or a type of '{query}'."
        # llm_template = "Is '{value}' the same as or a type of '{query}'? Directly answer with 'Yes' or 'No'."
    reformat_template = "The value is the same as or a type of '{query}'."
    llm_template = "Is '{value}' the same as or a type of '{query}'? Directly answer with 'Yes' or 'No'."
    corpus = df[attribute].values.tolist()

    args.llm_template = llm_template

    start_time = time.time()
    bm25_index = BM25Index(corpus, args.dataset)
    print(f"Time for loading BM25 index: {time.time() - start_time:.4f}s")
    load_index_time = time.time() - start_time
    start_time = time.time()
    hnsw_index = HNSWIndex(corpus, args.dataset)
    print(f"Time for loading HNSW index: {time.time() - start_time:.4f}s")
    load_index_time += time.time() - start_time

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
        result = solve_query(
            query, attribute, corpus, args, answers, bm25_index, hnsw_index
        )
        result["answers"] = answers
        result["time"] = time.time() - start_time
        results.append(result)
        save_results(results, output_path)
        cnt += 1
        if args.exp_name == "debug" and cnt >= 10:
            break
    # save results
    save_results(results, output_path)
