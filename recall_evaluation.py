import os
from load_data import load_data
import numpy as np
import json
import sys
import time


def print_case(corpus, query, pred, gt, rec):
    print(f"Query: {query}")
    print(f"Recall: {rec * 100:.2f}%")
    if type(gt[0]) == int:
        gt = [corpus[i] for i in gt]
    if type(pred[0]) == int:
        pred = [corpus[i] for i in pred]
    for p in pred:
        if p in gt:
            label = 1
        else:
            label = 0
        print(f"[Label] {label} | [Prediction] {p}")
    print("-" * 50)


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    result_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    answer_key = sys.argv[3] if len(sys.argv) > 3 else "pred"
    df, query_answer, query_template, filename = load_data(dataset_name)
    result_data = json.load(open(f"{result_dir}/{dataset_name}.json", "r"))

    if dataset_name == "paper":
        corpus = df["abstracts"].values.tolist()
        batch_size = 1024
    elif dataset_name == "product":
        corpus = df["Product_Title"].values.tolist()
        batch_size = 512
    else:
        cols = df.columns
        corpus = df[cols[0]].values.tolist()
        batch_size = 128

    k_list = [10, 20, 50, 100, 200, 300, 400, 500, 750, 1000]
    # k_list = [20]
    rec_dict = {}
    for k in k_list:
        rec_dict[k] = []
    case_cnt = 0
    for query, result in result_data.items():
        gt = query_answer[query]
        pred = result_data[query][answer_key]
        pred_bm25 = result["bm25_result"]
        pred_hnsw = result["hnsw_result"]
        if len(gt) == 0:
            continue
        if type(gt[0]) == str and type(pred[0]) == int:
            pred = [corpus[i] for i in pred]
            pred_bm25 = [corpus[i] for i in pred_bm25]
            pred_hnsw = [corpus[i] for i in pred_hnsw]
        elif type(gt[0]) == int and type(pred[0]) == str:
            gt = [corpus[i] for i in gt]
        # check the recall
        for k in k_list:
            pred_top_k = pred[:k]
            pred_hnsw = pred[:k]
            pred_top_k = pred_bm25[: k // 2]
            for i in range(k):
                if pred_hnsw[i] not in pred_top_k:
                    pred_top_k.append(pred_hnsw[i])
                    if len(pred_top_k) >= k:
                        break
            rec = len(set(pred_top_k) & set(gt)) / min(len(gt), k)
            rec_dict[k].append(rec)
        #     if rec < 0.5 and case_cnt < 10:
        #         print_case(corpus, query, pred_top_k, gt, rec)
        #         case_cnt += 1
        # if case_cnt >= 10:
        #     print("Too many cases, stop printing.")
        #     break
    print(f"Results for {dataset_name}:")
    for k in k_list:
        rec = np.mean(rec_dict[k])
        print(f"Recall@{k}: {rec * 100:.2f}")
