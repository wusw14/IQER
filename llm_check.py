import openai
from typing import List
from concurrent.futures import ThreadPoolExecutor
from appl import ppl, gen, SystemMessage, convo, records, SystemRole
from appl.compositor import Tagged, NumberedList, DashList
from reformulate import refine_query

client = openai.OpenAI(
    base_url="http://localhost:1117/v1",  # vLLM server address
    api_key="llama",  # dummy token
)

model_path = "meta-llama/Llama-3.3-70B-Instruct"


def process_single_prompt(cond: str, col: str, val: str) -> str:
    # replace the placeholder {col} with val
    statement = cond.replace(f"{{{col}}}", val)
    content = f"Please check if the statement '{statement}' is correct. Respond only with 'True' or 'False'."
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": content},
    # ]
    # return messages
    return content


def llm_check(query: str, corpus: list, template: str, checked_results={}) -> list:
    prompts = []
    corpus_new = []
    for value in corpus:
        if value in checked_results:
            continue
        corpus_new.append(value)
        prompt = template.format(value=value, query=query)
        # prompt += " Please directly answer with 'Yes' or 'No'."
        prompts.append(prompt)
    if len(prompts) == 0:
        return []
    results = run_inference(client, model_path, prompts, max_tokens=5)
    filtered_vals = []
    for val, result in zip(corpus_new, results):
        if result.strip().lower().startswith("yes"):
            filtered_vals.append(val)
    return filtered_vals


def llm_check_batch(cond: str, unique_vals: list) -> list:
    # split the unique_vals into batches of 20
    batches = [unique_vals[i : i + 20] for i in range(0, len(unique_vals), 20)]
    results = []
    for batch in batches:
        refined_results = refine_query(cond, batch)
        results.extend(refined_results)
    return results


# def llm_check(cond: str, col: str, unique_vals: list) -> list:
#     prompts = [process_single_prompt(cond, col, val) for val in unique_vals]
#     results = run_inference(client, model_path, prompts, max_tokens=5)
#     filtered_vals = []
#     for val, result in zip(unique_vals, results):
#         if result.strip().lower().startswith("true"):
#             filtered_vals.append(val)
#     return filtered_vals


@ppl
def appl_gen(prompt):
    prompt
    return gen()


def run_inference(
    client: openai.OpenAI,
    model_path: str,
    prompts: List[str],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    top_p: float = 0.8,
) -> List[str]:

    # def generate_completion(messages):
    #     response = client.chat.completions.create(
    #         model=model_path,
    #         messages=messages,
    #         max_tokens=max_tokens,
    #         temperature=temperature,
    #         top_p=0.8,
    #         presence_penalty=1.5,
    #         extra_body={
    #             "top_k": 20,
    #             "chat_template_kwargs": {"enable_thinking": False},
    #         },
    #     )
    #     return response.choices[0].message.content

    def generate_completion(prompt):
        ans = appl_gen(prompt)
        ans = str(ans).strip()
        return ans

    with ThreadPoolExecutor(max_workers=10) as executor:
        completions = list(executor.map(generate_completion, prompts))
    return completions
