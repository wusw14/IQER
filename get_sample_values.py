from load_data import load_data
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import sys
from sklearn.cluster import KMeans
import random

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]


if __name__ == "__main__":
    dataset = sys.argv[1]
    df, query_answer, query_template, path = load_data(dataset)
    if dataset == "paper":
        batch_size = 1024
    elif dataset == "product":
        batch_size = 512
    else:
        batch_size = 128
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    emb_model = SentenceTransformer(model_name)
    # get sample values for each column
    col_vals = {}
    for col in df.columns:
        # if the column is not a string, random sample 5 values
        if type(df[col].values[0]) != str:
            sample_values = df[col].sample(5).tolist()
        else:
            # cluster the values based on the embeddings
            vals = df[col].values.tolist()
            vals_embeddings = emb_model.encode(vals, batch_size=batch_size)
            vals_embeddings = vals_embeddings / np.linalg.norm(vals_embeddings, axis=1, keepdims=True)
            vals_embeddings = vals_embeddings.tolist()
            # cluster the values into 5 clusters
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(vals_embeddings)
            clusters = kmeans.labels_
            # sample 1 value from each cluster
            sample_values = []
            for i in range(5):
                cluster_vals = [vals[j] for j in range(len(vals)) if clusters[j] == i]
                sample_values.append(random.choice(cluster_vals))
        col_vals[col] = sample_values
    with open(f"{path}/{dataset}_sample_values.json", "w") as f:
        json.dump(col_vals, f, indent=4)
