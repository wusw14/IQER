# from lotus.models import OpenAIModel
import lotus
from lotus.models import LM, SentenceTransformersRM
import argparse
import pandas as pd
import json
import os
from load_data import load_data
import time
from lotus.types import CascadeArgs, ProxyModel
from lotus.vector_store import FaissVS


def save_results(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


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
    parser.add_argument(
        "--index_combine_method",
        type=str,
        choices=["weighted", "merge"],
        default="weighted",
    )
    parser.add_argument("--exp_name", type=str, default="debug")
    return parser.parse_args()


if __name__ == "__main__":
    # lm = OpenAIModel(
    #     model="meta-llama/Llama-3.3-70B-Instruct",
    #     api_base="http://localhost:1117/v1",
    #     api_key="llama",
    #     provider="vllm",
    #     max_tokens=1024,
    #     max_batch_size=512,
    # )
    lm = LM(
        model="openai/meta-llama/Llama-3.3-70B-Instruct",
        api_base="http://localhost:1118/v1",
        api_key="llama",
        max_tokens=1024,
        max_batch_size=512,
    )
    helper_lm = LM(
        model="openai/meta-llama/Llama-3.1-8B-Instruct",
        api_base="http://localhost:1110/v1",
        api_key="llama",
        max_tokens=10,
        max_batch_size=512,
    )
    # rm = SentenceTransformersRM(
    #     model="all-MiniLM-L6-v2",
    #     max_batch_size=512,
    #     device="cuda",
    # )
    # print(rm.transformer.device)
    vs = FaissVS()
    cascade_args = CascadeArgs(
        recall_target=0.95,
        precision_target=0.95,
        sampling_percentage=0.1,
        failure_probability=0.2,
        # proxy_model=ProxyModel.EMBEDDING_MODEL,
        proxy_model=ProxyModel.HELPER_LM,
    )
    # lotus.settings.configure(lm=lm, rm=rm, vs=vs)
    lotus.settings.configure(lm=lm, helper_lm=helper_lm)
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
        prefix = "The '{" + attribute + "}' " + "is the same as or a type of"
    else:
        cols = df.columns
        batch_size = 128
        attribute = cols[0]
        reformat_template = (
            f"According to the {attribute} name, the {attribute}"
            + " is the same as or a type of '{query}'."
        )
        prefix = "The '{" + attribute + "}' " + "is the same as or a type of"
    corpus = df[attribute].values.tolist()
    print(f"corpus size: {len(corpus)}")
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
        instruction = prefix + f" '{query}'."
        print(instruction)
        df_result = df.sem_filter(instruction, cascade_args=cascade_args)
        # df_result = df.sem_filter(instruction)
        result = {"query": query, "pred": df_result[attribute].values.tolist()}
        result["answers"] = answers
        result["time"] = time.time() - start_time
        results.append(result)
        print(result)
        save_results(results, output_path)
        cnt += 1
        if args.exp_name == "debug" and cnt >= 10:
            break
    # save results
    save_results(results, output_path)
