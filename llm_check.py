import openai
from typing import List
from concurrent.futures import ThreadPoolExecutor
from appl import ppl, gen, SystemMessage, convo, records, SystemRole
from appl.compositor import Tagged, NumberedList, DashList

# from reformulate import refine_query

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


def llm_check(
    query: str, corpus: list, template: str, checked_results={}, max_tokens=1024
) -> list:
    prompts = []
    corpus_new = []
    corpus = list(set(corpus))
    for value in corpus:
        if value in checked_results:
            continue
        corpus_new.append(value)
        prompt = template.format(value=value, query=query)
        # prompt += " Please directly answer with 'Yes' or 'No'."
        prompts.append(prompt)
    if len(prompts) == 0:
        return []
    results = run_inference(prompts, max_tokens)
    filtered_vals = []
    for val, result in zip(corpus_new, results):
        if result.strip().lower().startswith("yes"):
            filtered_vals.append(val)
    return filtered_vals


@ppl
def appl_gen(prompt, max_tokens=1024):
    prompt
    return gen(max_tokens=max_tokens)


def run_inference(prompts: List[str], max_tokens=1024) -> List[str]:

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
        ans = appl_gen(prompt, max_tokens)
        ans = str(ans).strip()
        return ans

    with ThreadPoolExecutor(max_workers=min(100, len(prompts))) as executor:
        completions = list(executor.map(generate_completion, prompts))
    return completions
