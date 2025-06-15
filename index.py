from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import hnswlib
from sentence_transformers import SentenceTransformer
import os
from load_data import load_data
import numpy as np
import json
import sys
import time
from retrieve import bm25_query, hnsw_query


# nltk.download("stopwords")
# nltk.download("punkt")


def build_bm25_index(corpus, dataset_name):
    # tokenization
    stop_words = set(stopwords.words("english"))
    tokenized_corpus = []
    for doc in corpus:
        tokens = word_tokenize(doc.lower())
        tokens = [token for token in tokens if token not in stop_words]
        tokenized_corpus.append(tokens)

    # build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    # save index
    with open(f"index/{dataset_name}_bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)


def build_hnsw_index(corpus, dataset_name):
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    embs = SentenceTransformer(emb_model).encode(corpus, batch_size=512)
    dim = embs.shape[1]
    num_elements = embs.shape[0]
    print(f"num elements: {num_elements}, dim: {dim}")

    # Declaring index
    p = hnswlib.Index(space="cosine", dim=dim)  # possible options are l2, cosine or ip
    p.init_index(max_elements=num_elements, ef_construction=200, M=40)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    p.set_ef(220)

    # Set number of threads used during batch search/construction
    # By default using all available cores
    # p.set_num_threads(4)
    p.add_items(embs)

    # Serializing and deleting the index:
    index_path = f"index/{dataset_name}_hnsw_index.bin"
    print("Saving index to '%s'" % index_path)
    p.save_index(index_path)


# def build_colbert_index(corpus, dataset_name):
#     # transform corpus to tsv format if not exits
#     if not os.path.exists(f"data/{dataset_name}.tsv"):
#         with open(f"data/{dataset_name}.tsv", "w") as f:
#             for i, doc in enumerate(corpus):
#                 f.write(f"{i}\t{doc}\n")
#     with Run().context(RunConfig(nranks=1, experiment=dataset_name)):
#         config = ColBERTConfig(nbits=2, root=dataset_name)
#         indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
#         indexer.index(name=dataset_name, collection=f"data/{dataset_name}.tsv")


def eval_recall(queries, results, query_answer, corpus):
    k_list = [10, 20, 50, 100, 200, 300, 400, 500, 750, 1000]
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

    # # Build BM25 index
    # if not os.path.exists(f"index/{dataset_name}_bm25_index.pkl"):
    #     print("Building BM25 index...")
    #     build_bm25_index(corpus, dataset_name)

    # # Build HNSW index
    # if not os.path.exists(f"index/{dataset_name}_hnsw_index.bin"):
    #     print("Building HNSW index...")
    #     build_hnsw_index(corpus, dataset_name)

    # # Build ColBERT index
    # if not os.path.exists(f"experiments/{dataset_name}"):
    #     print("Building ColBERT index...")
    #     build_colbert_index(corpus, dataset_name)

    # Query using BM25
    start_time = time.time()
    if index_type == "bm25":
        results, scores = bm25_query(corpus, query, dataset_name, top_n=1000)
    elif index_type == "hnsw":
        results, scores = hnsw_query(corpus, query, dataset_name, top_n=1000)
    time_cost = (time.time() - start_time) / len(query)
    print(f"Query time per query: {time_cost:.4f} seconds")

    # eval recall
    eval_recall(query, results, query_answer, corpus)

    # threshold range
    scores = np.array(scores)
    if index_type == "hnsw":
        scores = scores + 1
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
        f1_list = np.array(f1_list)
        k_list = np.array(k_list)
        pre_list = np.array(pre_list)
        rec_list = np.array(rec_list)
        if np.mean(f1_list) > opt_f1:
            opt_f1 = np.mean(f1_list)
            opt_threshold = threshold
            opt_k = np.mean(k_list)
            opt_pre = np.mean(pre_list)
            opt_rec = np.mean(rec_list)
    print(f"Optimal threshold: {opt_threshold}, Optimal k: {opt_k:.2f}")
    print(f"Optimal pre/rec/f1: {opt_pre*100:.2f}/{opt_rec*100:.2f}/{opt_f1*100:.2f}")

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
