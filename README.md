# Bridging the Query-Data Semantic Gap: Iterative Query Expansion for LLM-Enhanced Database Retrieval

This is the source code of IQER, which improves retrieval performance through iterative query expansion. 

Our paper is submitted to VLDB 2026. 

### Hardware environment
Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz  
NVIDIA A100 80GB  
Note: the experiments do not require the same hardware environment.

## Datasets
We construct three datasets from public data sources to simulate the scenarios with the semantic gap between query and data.
The datasets used in this work are in the folder "dataset"

| Dataset           | Table Size | # Query | Avg. # Answers |
|-------------------|------------|---------|----------------|
| Chemical Compound | 2878       | 44      | 4.41           |
| Animal            | 6459       | 82      | 6.79           |
| Product           | 27809      | 155     | 2.56           |

## Run the pipeline
```
python -u main.py --dataset animal --exp_name qwen_B100 --budget 100 --k 100
```
You need to change the configurations for different LLM servers in llm_check.py file.  
In our implementation, we use vLLM to deploy the local LLM server.
```
client = openai.OpenAI(
    base_url="http://localhost:1172/v1",  # vLLM server address
    api_key="qwen-72b",  
)

model_path = "Qwen/Qwen2.5-72B-Instruct"
```

## Run the evaluation
```
# dataset exp_name
python evaluation.py animal qwen_B100
```