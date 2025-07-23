from load_data import load_data
import numpy as np
import sys
import time
from index import BM25Index, HNSWIndex


# nltk.download("stopwords")
# nltk.download("punkt")


def eval_recall(queries, results, query_answer, corpus):
    k_list = [10, 50, 100, 200, 500, 1000]
    # k_list = [20]
    rec_dict = {}
    for k in k_list:
        rec_dict[k] = []
    for query, result in zip(queries, results):
        gt = query_answer[query]
        pred = result
        if len(gt) == 0:
            continue
        if type(gt[0]) == str and type(pred[0]) == int:
            pred = [corpus[i] for i in pred]
        elif type(gt[0]) == int and type(pred[0]) == str:
            gt = [corpus[i] for i in gt]
        # check the recall
        for k in k_list:
            pred_top_k = pred[:k]
            rec = len(set(pred_top_k) & set(gt)) / min(len(gt), k)
            rec_dict[k].append(rec)
    for k in k_list:
        rec = np.mean(rec_dict[k])
        print(f"Recall@{k}: {rec * 100:.2f}")
    return rec_dict


def merge_results(bm25_result, bm25_score, hnsw_result, hnsw_score):
    bm25_obj_scores = {obj: score for obj, score in zip(bm25_result, bm25_score)}
    hnsw_obj_scores = {obj: score for obj, score in zip(hnsw_result, hnsw_score)}
    obj_scores = {}
    for obj in set(list(bm25_result)) | set(list(hnsw_result)):
        obj_scores[obj] = (
            bm25_obj_scores.get(obj, 0) + hnsw_obj_scores.get(obj, 0)
        ) / 2
    sorted_obj_scores = sorted(obj_scores.items(), key=lambda x: x[1], reverse=True)
    results = [obj for obj, _ in sorted_obj_scores]
    scores = [obj_scores[obj] for obj, _ in sorted_obj_scores]
    return results, scores


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    df, query_answer, query_template, filename = load_data(dataset_name)
    index_type = sys.argv[2]

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
    query = list(query_answer.keys())

    # Query using BM25
    if index_type == "bm25":
        index = BM25Index(corpus, dataset_name)
        start_time = time.time()
        results, scores, _, _ = index.search(query, top_n=1000)
    elif index_type == "hnsw":
        index = HNSWIndex(corpus, dataset_name)
        start_time = time.time()
        results, scores, _, _, _ = index.search(query, top_n=1000)
    elif index_type == "hybrid":
        bm25_index = BM25Index(corpus, dataset_name)
        hnsw_index = HNSWIndex(corpus, dataset_name)
        start_time = time.time()
        bm25_results, bm25_scores, _, _ = bm25_index.search(query, top_n=1000)
        hnsw_results, hnsw_scores, _, _, _ = hnsw_index.search(query, top_n=1000)
        bm25_max, hnsw_max = max(np.max(bm25_scores), 1e-6), max(
            np.max(hnsw_scores), 1e-6
        )
        bm25_min, hnsw_min = max(np.min(bm25_scores), 0), max(np.min(hnsw_scores), 0)
        print(f"BM25 max: {bm25_max:.4f}, BM25 min: {bm25_min:.4f}")
        print(f"HNSW max: {hnsw_max:.4f}, HNSW min: {hnsw_min:.4f}")
        bm25_scores = np.array(bm25_scores)
        hnsw_scores = np.array(hnsw_scores)
        bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        hnsw_scores = (hnsw_scores - hnsw_min) / (hnsw_max - hnsw_min)
        results, scores = [], []
        for i, q in enumerate(query):
            bm25_result = bm25_results[i]
            hnsw_result = hnsw_results[i]
            bm25_score = bm25_scores[i]
            hnsw_score = hnsw_scores[i]
            result, score = merge_results(
                bm25_result, bm25_score, hnsw_result, hnsw_score
            )
            results.append(result)
            scores.append(score)
    time_cost = (time.time() - start_time) / len(query)
    print(f"Query time per query: {time_cost:.4f} seconds")

    # eval recall
    # eval_recall(query, results, query_answer, corpus)

    # threshold range
    if index_type != "hybrid":
        scores = np.array(scores)
        scores = scores / np.max(scores)
    opt_threshold = 0
    opt_f1 = 0
    for threshold in np.arange(0.1, 1.0, 0.01):
        f1_list, k_list = [], []
        pre_list, rec_list = [], []
        for i, q in enumerate(query):
            gt = query_answer[q]
            result = results[i]
            score = scores[i]
            # find the index of the result that is greater than the threshold
            index = np.where(score > threshold)[0]
            result = [result[i] for i in index]
            if len(gt) == 0:
                continue
            if len(result) == 0:
                f1_list.append(0)
                k_list.append(0)
                pre_list.append(0)
                rec_list.append(0)
                continue
            # align the result with ground truth
            if type(result[0]) == int and type(gt[0]) == str:
                result = [corpus[i] for i in result]
            # calculate f1 score
            tp = len(set(result) & set(gt))
            fp = len(set(result) - set(gt))
            fn = len(set(gt) - set(result))
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            f1_list.append(f1)
            k_list.append(len(result))
            pre_list.append(precision)
            rec_list.append(recall)
        # f1_list = np.array(f1_list)
        pre = np.mean(pre_list)
        rec = np.mean(rec_list)
        f1 = 2 * pre * rec / max(pre + rec, 1e-6)
        k = np.mean(k_list)
        if f1 > opt_f1:
            opt_f1 = f1
            opt_threshold = threshold
            opt_k = k
            opt_pre = pre
            opt_rec = rec
    print(f"Optimal threshold: {opt_threshold}, Optimal k: {opt_k:.2f}")
    print(f"Optimal pre/rec/f1: {opt_pre*100:.2f}/{opt_rec*100:.2f}/{opt_f1*100:.2f}")
    print(f"{opt_pre*100:.2f}")
    print(f"{opt_rec*100:.2f}")
    print(f"{opt_f1*100:.2f}")
    print(f"{opt_k:.2f}")
    print(f"{time_cost:.4f}")

    # # Save results
    # query_answer_new = {}
    # for i, q in enumerate(query):
    #     gt = query_answer[q]
    #     bm25_result = bm25_results[i]
    #     query_answer_new[q] = {
    #         "ground_truth": gt,
    #         "bm25_result": bm25_result,
    #     }
    # with open(f"index_result/{dataset_name}_bm25.json", "w") as f:
    #     json.dump(query_answer_new, f, indent=4)
    # print(f"BM25 query time per query: {bm25_time:.4f} seconds")

    # # Query using HNSW
    # start_time = time.time()
    # hnsw_results = hnsw_query(corpus, query, dataset_name, top_n=1000)
    # hnsw_time = (time.time() - start_time) / len(query)

    # # Query using ColBERT
    # start_time = time.time()
    # colbert_results = colbert_query(corpus, query, dataset_name, top_n=1000)
    # colbert_time = (time.time() - start_time) / len(query)

    # # Save results
    # query_answer_new = {}
    # for i, q in enumerate(query):
    #     gt = query_answer[q]
    #     bm25_result = bm25_results[i]
    #     hnsw_result = hnsw_results[i]
    #     colbert_result = colbert_results[i]
    #     query_answer_new[q] = {
    #         "ground_truth": gt,
    #         "bm25_result": bm25_result,
    #         "hnsw_result": hnsw_result,
    #         "colbert_result": colbert_result,
    #     }
    # with open(f"index_result/{dataset_name}.json", "w") as f:
    #     json.dump(query_answer_new, f, indent=4)
    # print(f"BM25 query time per query: {bm25_time:.4f} seconds")
    # print(f"HNSW query time per query: {hnsw_time:.4f} seconds")
    # print(f"ColBERT query time per query: {colbert_time:.4f} seconds")
