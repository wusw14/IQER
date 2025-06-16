from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import hnswlib
from sentence_transformers import SentenceTransformer
import os
import numpy as np


# nltk.download("stopwords")
# nltk.download("punkt")


class Index:
    def __init__(self, corpus: list[str], dataset_name: str):
        self.corpus = corpus
        self.dataset_name = dataset_name

    def build_index(self):
        pass

    def search(self, query: list[str], top_n: int = 1000):
        pass

    def agg_results(self, results: list[list[str]], scores: list[list[float]]):
        pass


class BM25Index(Index):
    def __init__(self, corpus: list[str], dataset_name: str):
        super().__init__(corpus, dataset_name)
        self.index_path = f"index/{dataset_name}_bm25_index.pkl"
        self.index = self.build_index()
        self.stop_words = set(stopwords.words("english"))

    def build_index(self) -> BM25Okapi:
        """
        Build BM25 index
        Load index from file if exists, otherwise build index and save to file
        """
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                return pickle.load(f)
        # tokenization
        tokenized_corpus = []
        for doc in self.corpus:
            tokens = word_tokenize(doc.lower())
            tokens = [token for token in tokens if token not in self.stop_words]
            tokenized_corpus.append(tokens)
        # build BM25 index
        bm25 = BM25Okapi(tokenized_corpus)
        # save index
        with open(self.index_path, "wb") as f:
            pickle.dump(bm25, f)
        return bm25

    def search(
        self, query: list[str], top_n: int = 1000
    ) -> tuple[list[list[int]], list[list[float]], list[list[float]]]:
        """
        Search BM25 index
        Return results, scores, full_scores
        """
        results = []
        scores = []
        full_scores = []
        for q in query:
            # tokenization
            tokenized_query = [
                token
                for token in word_tokenize(q.lower())
                if token not in self.stop_words
            ]
            score = self.index.get_scores(tokenized_query)
            full_scores.append(score)
            result = np.argsort(score)[::-1][:top_n].tolist()  # get top_n indices
            score = np.array(score)[result]
            results.append(result)
            scores.append(score)
        return results, scores, full_scores


class HNSWIndex:
    def __init__(self, corpus: list[str], dataset_name: str):
        self.corpus = corpus
        self.dataset_name = dataset_name
        self.emb_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.dim = 384
        self.sim_metric = "cosine"
        self.index_path = f"index/{dataset_name}_hnsw_index.bin"
        self.index = self.build_index()

    def build_index(self) -> hnswlib.Index:
        """
        Build HNSW index
        Load index from file if exists, otherwise build index and save to file
        """
        if os.path.exists(self.index_path):
            # load HNSW index
            p = hnswlib.Index(space=self.sim_metric, dim=self.dim)
            p.load_index(self.index_path)
            return p

        embs = SentenceTransformer(self.emb_model).encode(self.corpus, batch_size=512)
        num_elements = embs.shape[0]
        print(f"num elements: {num_elements}, dim: {self.dim}")

        # Declaring index
        p = hnswlib.Index(
            space="cosine", dim=self.dim
        )  # possible options are l2, cosine or ip
        p.init_index(max_elements=num_elements, ef_construction=200, M=40)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        p.set_ef(220)

        # Set number of threads used during batch search/construction
        # By default using all available cores
        # p.set_num_threads(4)
        p.add_items(embs)

        # Serializing and deleting the index:
        print("Saving index to '%s'" % self.index_path)
        p.save_index(self.index_path)
        return p

    def search(
        self, query: list[str], top_n: int = 1000
    ) -> tuple[list[list[int]], list[list[float]], np.ndarray, list[int], np.ndarray]:
        """
        Search HNSW index
        Return results, scores, query_embedding, unique_results, unique_results_embs
        """
        # encode query
        query = [q.lower() for q in query]
        query_embedding = SentenceTransformer(self.emb_model).encode(
            query, batch_size=512
        )
        # load HNSW index
        p = hnswlib.Index(space=self.sim_metric, dim=self.dim)
        p.load_index(self.index_path)
        # search for nearest neighbors
        labels, distances = p.knn_query(query_embedding, k=top_n)
        results = []
        scores = []
        unique_results = []
        for i, q in enumerate(query):
            result, score = [], []
            for j in range(top_n):
                doc_id = int(labels[i][j])
                result.append(doc_id)
                score.append(1 - distances[i][j])
            results.append(result)
            scores.append(score)
            unique_results.extend(result)
        unique_results = list(set(unique_results))
        # get the embeddings of the unique results
        unique_results_embs = p.get_items(unique_results, return_type="numpy")
        return results, scores, query_embedding, unique_results, unique_results_embs
