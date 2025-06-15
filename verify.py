import os
import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from utils import parse_response
from appl import ppl, gen, SystemMessage, convo, records, SystemRole
from appl.compositor import Tagged, NumberedList, DashList


def retrieve(vals: list, db: str, table: str, col: str, args) -> list:
    """
    Retrieve values from the database based on the generated values.

    Args:
        vals (list): The list of generated values.
        db (str): The database name.
        table (str): The table name.
        col (str): The column name.

    Returns:
        list: The retrieved values from the database.
    """
    # map vals to embeddings
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embs = emb_model.encode(vals)
    # normalize the embeddings
    query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    query_emb = np.mean(query_embs, axis=0)
    # load_embeddings or index
    if os.path.exists(f"{args.input_dir}/{db}/{table}/{col}/embs.index"):
        # Load the index
        p = hnswlib.Index(space="l2", dim=768)
        p.load_index(f"{args.input_dir}/{db}/{table}/{col}/embs.index")
        # Query the index
        ids, distances = p.knn_query([query_emb], k=args.k)
        ids = ids[0]
    else:
        embs = np.load(f"{args.input_dir}/{db}/{table}/{col}/embs.npy")
        # cal_similarity
        scores = np.dot(query_emb, embs.T)
        ids = np.argsort(scores)[::-1][: args.k]
    # load the values corresponding to embs
    cell_vals = json.load(
        open(f"{args.input_dir}/{db}/{table}/{col}/vals.json", "r", encoding="utf-8")
    )
    # get the values corresponding to the ids
    retrieved_vals = [cell_vals[i] for i in ids]
    return retrieved_vals


def filter_val(vals: list, cond: str, reverse: bool = False) -> list:
    """
    Filter values based on the condition.

    Args:
        vals (list): The list of values to filter.
        cond (str): The condition to filter by.
        reverse (bool): Whether to reverse the filtering.

    Returns:
        list: The filtered values.
    """
    # Placeholder for actual filtering logic
    # This should be replaced with the actual implementation
    ans = llm_filter(vals, cond, reverse=reverse)
    ans = parse_response(ans)
    filtered_vals = ans.split("|")
    filtered_ids = [int(v.strip()) for v in filtered_vals]
    return filtered_ids


@ppl
def llm_filter(vals: list, cond: str, reverse: bool = False):
    SystemMessage("You are a knowledeable and helpful assistant.")
    if reverse:
        "Task: Identify the values that do not meet the given condition and separate them from those that do."
    else:
        "Task: Identify the values that meet the given condition and separate them from those that do not."

    f"Condition: cond"

    "Values:"
    for i, v in enumerate(vals):
        f"{i+1}: {v}"

    "Output:"
    if reverse:
        "<thought>Analyze the values and the condition to determine which values do not meet the condition.</thought>"
    else:
        "<thought>Analyze the values and the condition to determine which values meet the condition.</thought>"
    "<answer>List the serial numbers of the filtered values​, separated by |</answer>"
    return gen()


def eval_alignment(cond: str, vals: list, db: str, table: str, col: str, args) -> bool:
    """
    Evaluate the alignment of the condition with the column and value.

    Args:
        cond (str): The condition to evaluate.
        vals (list): The list of generated values.
        db (str): The database name.
        table (str): The table name.
        col (str): The column name.

    Returns:
        bool: True if the generated values are good to retrieve values from DB, False otherwise.
    """
    # Step 1: retrieve values based on generated values from DB
    retrieved_values = retrieve(vals, db, table, col, args)
    # Step 2: check if the retrieved values are aligned with the original condition
    candidate_ids = [i + 1 for i in range(len(retrieved_values))]
    score = 0
    filtered_rank = []
    for _ in range(5):
        vals = [retrieved_values[i] for i in candidate_ids]
        filtered_ids = filter_val(vals, cond, reverse=False)
        complement_ids = filter_val(vals, cond, reverse=True)
        full_set = set(list(range(len(vals))))
        conflict_set = set(filtered_ids) & set(complement_ids)
        ignore_set = full_set - (set(filtered_ids) | set(complement_ids))
        filtered_set = set(filtered_ids) - set(complement_ids)
        filtered_rank.extend([candidate_ids[i] for i in filtered_set])
        cur_score = len(conflict_set + ignore_set)
        if cur_score == 0:
            break
        candidate_ids = [candidate_ids[i] for i in conflict_set + ignore_set]
        score += cur_score

    return True
