from utils import sample_table, parse_response
from appl import ppl, gen, SystemMessage, convo, records, SystemRole
from appl.compositor import Tagged, NumberedList, DashList
import numpy as np


@ppl
def gen_reformulate(cond: str, col: str, value: list, history: list = []):
    SystemMessage(
        "You are a data expert skilled in semantic interpretation and pattern matching for database systems."
    )

    "Task: Generate values that match both the query condition and the formatting style of a given database table column."

    "Generate 2-5 possible values that:"
    with NumberedList():
        "Are consistent with the format/style of the example values."
        "Satisfy the specified query condition."
        "Vary realistically (avoid simple patterns)."
        "If you know the answer for the condition, please provide the answer directly. If you are not sure about the answer, please provide a list of possible values that could satisfy the condition."

    "## Examples"
    "Original Condition: Country in Southeast Asia"
    "Column: country"
    "Sample Values: ['China', 'India', 'Japan', 'Canada', 'France']"
    "Output:"
    "<thought>The condition requires countries geographically in Southeast Asia. The samples show proper noun country names in title case. Must generate only sovereign nations from this region in identical format.</thought>"
    "<answer>Brunei | Cambodia | Indonesia | Laos | Malaysia | Myanmar | Philippines | Singapore | Thailand | Vietnam</answer>"

    "\nPlease analyze the following condition and provide the output in the same format as the examples."
    "Original Condition: " + cond
    "Column: " + col
    f"Sample Values: {value}"
    if len(history) > 0:
        "Prvious Generated Values:"
        for h in history:
            f"{h}"
        "The previous generated values are not suitable. Please regenerate more suitable values."
    "Output:"
    "<thought>Analyze both the semantic intent of the condition AND all observable formatting patterns in the samples.</thought>"
    "<answer>Generate values here, separated by |</answer>"

    return gen()


def reformulate(cond: str, col: str, value: list, history: list = []) -> str:
    """
    Reformulate the condition based on the column and value.

    Args:
        cond (str): The original condition.
        col (str): The column name.
        value (list): The list of values.

    Returns:
        str: The reformulated condition.
    """
    # sample_vals = np.random.choice(value, 5, replace=False)
    sample_vals = value
    if len(sample_vals) > 5:
        sample_vals = np.random.choice(sample_vals, 5, replace=False)
    ans = gen_reformulate(cond, col, sample_vals, history=history)
    ans = parse_response(ans, "answer")
    values = ans.split("|")
    values = [v.strip() for v in values]
    return values


@ppl
def gen_refine(condition: str, query_list: list, attempt: int = 0):
    SystemMessage(
        "You are an expert in determining whether the values satisfy the conditon."
    )

    "Your task is to select the values that satisfy the condition."
    # "Briefly analyze the values and determine if they satisfy the condition."
    # "Your output should be only a json object with the following format:"
    # "{'valid_values': [value1, ...], 'invalid_values': [value2, ...]}"

    "Please analyze the following condition and values and determine if the values satisfy the condition."
    # "First, reason about the question clearly but briefly in one or two sentences."
    # "Then, provide the final answer based on your reasoning."
    # "Only output the minimal necessary thought process."
    # "Answer the question in your mind, but only write the minimum needed to justify your conclusion. Avoid unnecessary elaboration."
    "You should only output a list of values that satisfy the condition with the following format: ['value1', 'value2', ...]."
    "Do not output any other text except the list of values."
    if attempt > 0:
        f"Your previous {attempt} attempts do not follow the format. Please try again."
    f"Condition: {condition}"
    f"Values: {query_list}"
    "The list of values that satisfy the condition is:"
    return gen()


def refine_query(condition: str, query_list: list) -> list:
    """
    Refine the query list based on the query.
    """
    attempt = 0
    while True:
        ans = gen_refine(condition, query_list, attempt)
        ans = str(ans)
        print(f"LLM Refine: {ans}")
        # find the json object in the answer
        start_idx = ans.rfind("[")
        end_idx = ans.rfind("]") + 1
        try:
            query_list = eval(ans[start_idx:end_idx])
            break
        except Exception as e:
            print(f"Error: {e}")
            attempt += 1
            if attempt > 3:
                return None
    return query_list
    # json_str = ans[start_idx:end_idx]
    # try:
    #     json_obj = eval(json_str)
    #     return json_obj["valid_values"]
    # except Exception as e:
    #     print(f"Error: {e}")
    #     return query_list


@ppl
def gen_score(condition: str, query_list: list):
    SystemMessage("You are an expert in scoring the query words in the query list.")

    "Given the condition and a list of values, assign each value a score from 0 to 2 based on whether it satisfies the condition:"
    with DashList():
        "0: Not satisfies the condition"
        "1: Not sure"
        "2: Satisfies the condition"

    "Return only a JSON object in this format: {'value1': score1, 'value2': score2, ...}."
    "Do not include any additional text or explanation."
    f"Condition: {condition}"
    f"Values: {query_list}"
    "JSON Output:"
    return gen()


def score_query(condition: str, query_list: list) -> list:
    """
    Score the query words in the query list.
    """
    attempt = 0
    while True:
        ans = gen_score(condition, query_list)
        ans = str(ans)
        try:
            score_dict = eval(ans)
            print(f"Score: {score_dict}")
            return score_dict
        except Exception as e:
            print(f"Error: {e}")
            attempt += 1
            if attempt > 3:
                break
    return {q: 1 for q in query_list}
