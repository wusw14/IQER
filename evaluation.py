import os
from load_data import load_data
import numpy as np
import json
import sys
import time
import re
from utils import cal_ndcg


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
    exp_name = sys.argv[2]
    method = sys.argv[3] if len(sys.argv) > 3 else "weighted"
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    df, query_answer, query_template, filename = load_data(dataset_name)
    result_data = json.load(
        open(f"results/{exp_name}/{dataset_name}_{method}.json", "r")
    )
    # result_data = {d["query"]: d["pred"] for d in result_data}
    # gt_data = json.load(open(f"{result_dir}/{method}/{dataset_name}_refined.json", "r"))
    # gt_data = {d["query"]: d["answers"] for d in gt_data}

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

    pre_list, rec_list, f1_list = [], [], []
    k_list = []
    time_list = []
    retrieve_recall_list = []
    ndcg_list = []
    # for d in gt_data:
    cnt = 0
    allowed_chars = r"a-zA-Z0-9 \+\-\&_\.\(\)"
    for d in result_data:
        if limit is not None and cnt >= limit:
            break
        cnt += 1
        query = d["query"]
        # gt = d["answers"]
        gt = query_answer.get(query, [])
        # gt = gt_data.get(query, [])
        # pred = result_data[query]
        pred = d["pred"]
        time_list.append(d.get("time", 0))
        retrieved = d.get("retrieved", pred)
        query_scores = d.get("query_scores", {})
        if len(gt) == 0:
            continue
        k = len(retrieved)
        k_list.append(k)
        pred_org = list(pred)
        pred = []
        for p in pred_org:
            score = query_scores.get(p, 2)
            if score > 0:
                pred.append(p)
        if len(pred) == 0:
            pred = pred_org
        # pred = pred_org
        if len(pred) == 0:
            precision, recall, f1 = 0, 0, 0
        else:
            if type(gt[0]) == str and type(pred[0]) == int:
                pred = [corpus[i] for i in pred]
                retrieved = [corpus[i] for i in retrieved]
            elif type(gt[0]) == int and type(pred[0]) == str:
                gt = [corpus[i] for i in gt]
            gt = [re.sub(allowed_chars, " ", s) for s in gt]
            pred = [re.sub(allowed_chars, " ", s) for s in pred]
            retrieved = [re.sub(allowed_chars, " ", s) for s in retrieved]
            # calculate precision, recall, f1
            tp = len(set(pred) & set(gt))
            precision = tp / len(pred)
            recall = tp / len(gt)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        retrieve_recall = len(set(retrieved) & set(gt)) / len(gt)
        ndcg = cal_ndcg(retrieved, gt)
        retrieve_recall_list.append(retrieve_recall)
        ndcg_list.append(ndcg)
        if retrieve_recall < 1:
            print(f"Query: {query}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"Retrieved Recall: {retrieve_recall:.4f}")
            print(f"NDCG: {ndcg:.4f}")
            print(f"Pred: {pred}")
            print(f"GT: {gt}")
            print("==========================")
        pre_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)
    avg_pre = np.mean(pre_list) * 100
    avg_rec = np.mean(rec_list) * 100
    avg_f1 = 2 * avg_pre * avg_rec / max(avg_pre + avg_rec, 1e-6)
    avg_ndcg = np.mean(ndcg_list) * 100
    print(f"pre: {avg_pre:.2f}")
    print(f"rec: {avg_rec:.2f}")
    print(f"f1: {avg_f1:.2f}")
    print(f"Average_k: {np.mean(k_list):.2f}")
    print(f"Average_retrieve_recall: {np.mean(retrieve_recall_list) * 100:.2f}")
    print(f"Average_ndcg: {avg_ndcg:.2f}")
    print(f"Average_time: {np.median(time_list):.2f}")
