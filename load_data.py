import pandas as pd
import json


def load_data(dataset_name):
    if dataset_name == "paper":
        folder = "../dataset/paper"
        df = pd.read_csv(f"{folder}/paper.csv")
        query_answer = json.load(open(f"{folder}/query_answer.json"))
        query_template = "Find the paper ids of all the papers about {value}"
    elif dataset_name == "product":
        folder = "../dataset/product"
        df = pd.read_csv(f"{folder}/product.csv")
        query_answer = json.load(open("results/llm/product_refined.json"))
        query_answer = {x["query"]: x["answers"] for x in query_answer}
        query_answer_org = json.load(open(f"{folder}/query_answer.json"))
        for key, value in query_answer_org.items():
            if key not in query_answer:
                query_answer[key] = value
        query_template = "Find all the products that are {value}"
    elif dataset_name in [
        "animal",
        "plant",
        "chemical_compound",
        "transportation_device",
    ]:
        folder = "../dataset/ConceptNet"
        df = pd.read_csv(f"{folder}/{dataset_name}_candidates.csv")
        df = df.drop_duplicates()
        query_answer = json.load(open(f"results/llm/{dataset_name}_refined.json"))
        query_answer = {x["query"]: x["answers"] for x in query_answer}
        query_template = f"Find all the {dataset_name} that are " + "{value}"
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    return df, query_answer, query_template, folder
