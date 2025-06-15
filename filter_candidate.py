import argparse
import pandas as pd
import json
import os
from llm_check import run_inference
from load_data import load_data
import openai


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


if __name__ == "__main__":
    args = parse_args()
    print(args)

    client = openai.OpenAI(
        base_url="http://localhost:1118/v1",  # vLLM server address
        api_key="llama",  # dummy token
    )

    model_path = "meta-llama/Llama-3.3-70B-Instruct"

    df, query_answer, query_template, path = load_data(args.dataset)
    candidates = df.values[
        :, 0
    ].tolist()  # assuming the first column contains the candidates
    print(len(candidates), "candidates found.")
    all_prompts = []
    for val in candidates:
        prompt = f"Is {val} a type of {args.dataset}? Directly answer with 'Yes/No/Unsure' and confidence score (0 to 1) without any redundant words."
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        all_prompts.append(messages)
    results = run_inference(
        client=client, model_path=model_path, prompts=all_prompts, max_tokens=10
    )
    filtered_gt = []
    for val, res in zip(candidates, results):
        if res.startswith("Yes"):
            filtered_gt.append(val)

    df_new = pd.DataFrame({args.dataset: filtered_gt})
    df_new.to_csv(f"{path}/{args.dataset}_candidates.csv", index=False)
    print(f"Original/Filtered: {len(df)}/{len(df_new)}")
