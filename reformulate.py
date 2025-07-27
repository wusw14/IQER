# from appl import ppl, gen, SystemMessage, convo, records, SystemRole
# from appl.compositor import Tagged, NumberedList, DashList
from llm_check import run_inference


# @ppl
# def gen_reformulate(cond: str, col: str, value: list, history: list = []):
#     SystemMessage(
#         "You are an expert database assistant specializing in query expansion and semantic matching."
#     )

#     "Your task is to generate semantically related search terms based on:"
#     with NumberedList():
#         "A user's query input (which may contain synonyms, partial matches, or hierarchical concepts)"
#         "A list of sample values from a database column (to maintain format consistency)"

#     "Instructions:"
#     with NumberedList():
#         "Analyze the query's core concept and identify:"
#         with DashList(indent=2):
#             "Conceptual synonyms and alternative terminology"
#             "Narrower terms (more specific instances/subtypes)"
#         "Use the example values as format references to maintain consistency with database conventions"
#         "Generate 2-10 expanded and diversified search terms"
#         "Exclude exact duplicates of the original query term and historical generated values if any"
#         "Prioritize terms that would actually appear in database records"

#     "Input:"
#     "Original Query Term: " + cond
#     "Column: " + col
#     f"Sample Values: {value}"
#     if len(history) > 0:
#         "Previously Generated Values:"
#         for h in history:
#             f"{h}"

#     "Output: please directly output the generated search terms separated by ' | ' without any other text. If no more terms can be generated, respond with 'None'."
#     return gen()


def get_reformulate_prompt(
    cond: str, col: str, value: list, history: list = [], reform_type: str = "zero-shot"
):
    system_prompt = "You are an expert database assistant specializing in query expansion and semantic matching."
    user_prompt = f"""Your task is to generate semantically related search terms based on:
1. A user's query input (which may contain synonyms, abbreviations, partial matches, or hierarchical concepts)
2. A list of sample values from a database column (to maintain format consistency)

Instructions:
1. Analyze the query's core concept and identify:
    - Conceptual synonyms, abbreviations, and alternative terminology
    - Narrower terms (more specific instances/subtypes)
2. Use the example values as format references to maintain consistency with database conventions
3. Generate 2-10 expanded and diversified search terms
4. Exclude exact duplicates of the original query term and historical generated values if any
5. Prioritize terms that would actually appear in database records
"""
    if reform_type == "few-shot":
        user_prompt += f"""
Examples:
Input:
Original Query Term: coffee
Column: drink
Sample Values: ["cappuccino", "black tea", "water", "orange juice", "soda"]
Output: cappuccino | espresso | mocha | latte | americano | drip coffee | affogato

Input:
Original Query Term: Cantonese dim sum
Column: food
Sample Values: ["spaghetti carbonara", "chicken tikka masala", "beef bourguignon", "miso glazed salmon", "vegetable paella"]
Output: ["shrimp dumpling", "barbecue pork bun", "rice noodle roll", "chicken feet", "turnip cake", "taro dumpling", "egg tart"]

Now you are given the following input:
"""

    user_prompt += f"""
Input:
Original Query Term: {cond}
Column: {col}
Sample Values: {value}

"""
    if len(history) > 0:
        history_str = " | ".join([f"{h}" for h in history])
        user_prompt += f"Previously Generated Values:\n{history_str}\n"
    if reform_type == "cot":
        user_prompt += f"""Your output should follow this format:
<think>Your thinking process for generating the search terms</think>
<output>The generated search terms separated by ' | '</output>
"""
    else:
        user_prompt += "Output: please directly output the generated search terms separated by ' | ' without any other text. If no more terms can be generated, respond with 'None'."
    return system_prompt, user_prompt


def reformulate(
    cond: str, col: str, value: list, history: list = [], reform_type: str = "zero-shot"
) -> str:
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
    # if len(sample_vals) > 5:
    #     sample_vals = np.random.choice(sample_vals, 5, replace=False)
    print(f"Query Condition: {cond}")
    system_prompt, user_prompt = get_reformulate_prompt(
        cond, col, sample_vals, history, reform_type
    )
    # ans = gen_reformulate(cond, col, sample_vals, history=history)
    ans = run_inference([user_prompt], system_prompts=[system_prompt])
    ans = ans[0]
    print(f"LLM Reformulate: {ans}")
    if reform_type == "cot":
        ans = ans.split("<output>")[1].split("</output>")[0].strip()
    values = str(ans).split("|")
    if "None" in values:
        return []
    values = [v.split("\n")[0].strip() for v in values if len(v.strip()) > 0]
    return values


# @ppl
# def gen_score(condition: str, query_list: list):
#     SystemMessage("You are an expert in scoring the query words in the query list.")

#     "Given the condition and a list of values, assign each value a score from 0 to 2 based on whether it satisfies the condition:"
#     with DashList():
#         "0: Not satisfies the condition"
#         "1: Not sure"
#         "2: Satisfies the condition"

#     "Return only a JSON object in this format: {'value1': score1, 'value2': score2, ...}."
#     "Do not include any additional text or explanation."
#     f"Condition: {condition}"
#     f"Values: {query_list}"
#     "JSON Output:"
#     return gen()


def score_query(condition: str, query_list: list) -> list:
    """
    Score the query words in the query list.
    """
    # check each query in the query list: zero-shot check if the query satisfies the condition
    if len(query_list) == 0:
        return {}
    prompts = []
    for q in query_list:
        prompt = f"Please check if the value '{q}' satisfies the condition '{condition}'. Directly answer with 'Yes', 'No' or 'Unsure' without any other text."
        prompts.append(prompt)
    results = run_inference(prompts, max_tokens=1)
    score_dict = {}
    for q, result in zip(query_list, results):
        if result.strip().lower().startswith("yes"):
            score_dict[q] = 2
        elif result.strip().lower().startswith("no"):
            score_dict[q] = 0
        else:
            score_dict[q] = 1
    return score_dict
