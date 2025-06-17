import argparse
import pandas as pd
import json
import os
from load_data import load_data
import time
from reformulate import reformulate, score_query
from retrieve import retrieve_corpus
from iterative_check import iterative_check_retrieved_objs


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
    samples = pos_objs + neg_objs[: len(pos_objs)]
    return samples


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
    # Step 1: Initialization
    reformulate_time, refine_time, retrieve_time = 0, 0, 0
    pos_objs = []  # the positive objects
    checked_obj_dict = {}
    query_scores = {query: 2}
    last_iter_pos_objs = []
    generated_query_list = []
    print(f"======Processing Query: [{query}]=======")
    # Step 2: Iteratively refine the query and retrieve the cell values
    for step in range(args.steps):
        cur_iter_pos_objs = []
        print(f"======Step {step}=======")
        # Reformulate the query
        start_time = time.time()
        if step > 0:
            sample_values = get_sample_values(checked_obj_dict)
            cur_generated_query_list = reformulate(
                reformat_template.format(query=query),
                attribute,
                sample_values,
                generated_query_list,
            )
            generated_query_list.extend(cur_generated_query_list)
        else:
            cur_generated_query_list = []
        print(f"Reformulated query: {cur_generated_query_list}")
        reformulate_time += time.time() - start_time

        # score the query words in the query list
        start_time = time.time()
        new_query_objs = cur_generated_query_list + last_iter_pos_objs
        if len(new_query_objs) > 0:
            query_scores_new = score_query(
                reformat_template.format(query=query), new_query_objs
            )
            query_scores.update(query_scores_new)
        refine_time += time.time() - start_time

        # retrieve the corpus based on the query list
        start_time = time.time()
        query_list = [q for q, s in query_scores.items() if s > 1]
        print(f"Query list: {query_list}")
        retrieved_info = retrieve_corpus(query_scores, corpus, args, checked_obj_dict)
        retrieve_time += time.time() - start_time

        # iteratively examine the retrieved objs
        cur_iter_pos_objs, checked_obj_dict = iterative_check_retrieved_objs(
            query,
            retrieved_info,
            args,
            checked_obj_dict,
            step,
        )
        pos_objs.extend(cur_iter_pos_objs)
        last_iter_pos_objs = list(cur_iter_pos_objs)
        if len(cur_iter_pos_objs) == 0 and step > 0:
            break

    new_query_objs = last_iter_pos_objs
    if len(new_query_objs) > 0:
        query_scores_new = score_query(
            reformat_template.format(query=query), new_query_objs
        )
        query_scores.update(query_scores_new)

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
