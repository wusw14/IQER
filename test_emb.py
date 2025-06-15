from load_data import load_data
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]


if __name__ == "__main__":
    dataset = sys.argv[1]
    df, query_answer, query_template, folder = load_data(dataset)
    if dataset == "paper":
        candidates = df["abstracts"].values.tolist()
        batch_size = 1024
    elif dataset == "product":
        candidates = df["Product Title"].values.tolist()
        batch_size = 512
    else:
        cols = df.columns
        candidates = df[cols[0]].values.tolist()
        batch_size = 128
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    emb_model = SentenceTransformer(model_name)
    candidates_embeddings = emb_model.encode(candidates, batch_size=batch_size)

    rec_dict = {}
    k_list = [5, 10, 15, 20, 25, 50, 75, 100, 200, 300, 400, 500, 1000]
    for k in k_list:
        rec_dict[k] = []
    for query, answers in query_answer.items():
        keywords = query
        keyword_embedding = emb_model.encode(keywords)
        # Calculate the cosine similarity
        # Normalize the embeddings
        keyword_embedding = keyword_embedding / np.linalg.norm(keyword_embedding)
        candidates_embeddings = candidates_embeddings / np.linalg.norm(
            candidates_embeddings, axis=1, keepdims=True
        )
        cosine_similarity = np.dot(candidates_embeddings, keyword_embedding.T)
        # sort the candidates by cosine similarity
        sorted_ids, sorted_cands, _ = zip(
            *sorted(
                zip(list(range(len(candidates))), candidates, cosine_similarity),
                key=lambda x: x[-1],
                reverse=True,
            )
        )
        for k in k_list:
            if dataset == "paper":
                preds = sorted_ids[:k]
            else:
                preds = sorted_cands[:k]
            rec = len(set(preds) & set(answers)) / min(len(answers), k)
            rec_dict[k].append(rec)
    print("Result", dataset)
    for k in k_list:
        print(f"Rec@{k}: {np.mean(rec_dict[k])* 100:.2f}%")
