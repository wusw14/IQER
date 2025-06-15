import argparse
import pandas as pd
import json
import os
from parsing import parse_query
from reformulate import reformulate
from align import transform_to_sql
from verify import eval_alignment
from copy import deepcopy
from utils import execute_sql
from llm_check import llm_check
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../TAG-Bench")
    parser.add_argument("--dataset", type=str, default="TAG")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--test_type", type=str, default="dev")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5)
    return parser.parse_args()


def solve_query(data: dict, args) -> dict:
    # This is a placeholder function. Replace with actual query solving logic.
    # For example, if data is a dictionary, you might want to extract some information from it.
    query = data["query"]
    db = data["db"]
    table = data["table"]
    # Step 1: load sample values, which are pre-selected
    col_vals = json.load(open(f"{args.input_dir}/{db}/{table}.json"))
    # Step 2: schema alignment; query parsing
    cond2col = parse_query(query, db, table, col_vals)  # {cond: col}
    # Step 3: collect the conditions that are SQL clauses
    sql_conds, remain_conds, remain_cols = [], [], []
    for cond, col in cond2col.items():
        # check if the condition is a SQL clause
        sql_ops = ["=", ">", "<", ">=", "<=", "!=", "in", "not in", "is", "is not", "like", "not like", "between", "not between"]
        if any(op in cond for op in sql_ops):
            sql_conds.append(cond)
        else:
            remain_conds.append(cond)
            remain_cols.append(col)
    # combine the SQL clauses into a single SQL query
    sql_query = "SELECT " + ", ".join(remain_cols) + " FROM " + table + " WHERE " + " AND ".join(sql_conds)
    results = execute_sql(sql_query, db)
    df = pd.DataFrame(results, columns=remain_cols)
    # Step 4: let LLM check the values for each remaining condition
    result = {}
    # sort the conditions by the number of unique values
    remain_conds, remain_cols = zip(*sorted(zip(remain_conds, remain_cols), key=lambda x: len(df[x[1]].unique())))
    for cond, col in zip(remain_conds, remain_cols):
        unique_vals = df[col].unique()
        filtered_vals = llm_check(cond, col, unique_vals)
        result[col] = {"cond": cond, "org_vals": unique_vals.tolist(), "filtered_vals": filtered_vals.tolist()}
        df = df[df[col].isin(filtered_vals)]
    return {"id": data["id"], "result": result}


if __name__ == "__main__":
    args = parse_args()
    print(args)

    # load data
    data = json.load(os.path.join(args.input_dir, args.dataset, "data.json"))
    # solve query
    results = []
    for d in data:
        start_time = time.time()
        result = solve_query(d, args)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        result["time"] = f"{end_time - start_time:.2f}"
        results.append(result)
    # save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.dataset}.json")
    with open(output_path, "w") as f:
        json.dump(results, f)
