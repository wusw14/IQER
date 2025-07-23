# python -u eval_lotus.py --dataset chemical_compound --exp_name lotus_proxy --llm llama > logs/lotus_proxy/chemical_llama.log
# python -u eval_lotus.py --dataset animal --exp_name lotus_proxy --llm llama > logs/lotus_proxy/animal_llama.log
# python -u eval_lotus.py --dataset product --exp_name lotus_proxy --llm llama > logs/lotus_proxy/product_llama.log

# python -u main.py --dataset animal --index_combine_method weighted --exp_name ab_reform --steps 10 --rethink > logs/ab/animal_reform.log
# python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name ab_reform --steps 10 --rethink > logs/ab/chemical_reform.log
# python -u main.py --dataset product --index_combine_method weighted --exp_name ab_wo_tbval --steps 10 --rethink > logs/ab/product_wo_tbval.log

# python -u tablerag.py --dataset animal --alpha 0.5 --exp_name tablerag_0.5_B100_llama --budget 100 --k 100 > logs/tablerag/animal_0.5_B100_llama.log
# python -u tablerag.py --dataset chemical_compound --alpha 0.5 --exp_name tablerag_0.5_B100_llama --budget 100 --k 100 > logs/tablerag/chemical_0.5_B100_llama.log
# python -u tablerag.py --dataset product --alpha 0.5 --exp_name tablerag_0.5_B100_llama --budget 100 --k 100 > logs/tablerag/product_0.5_B100_llama.log

# python -u tablerag.py --dataset animal --alpha 0.5 --exp_name tablerag_0.5_B200_llama --budget 200 --k 200 > logs/tablerag/animal_0.5_B200_llama.log
# python -u tablerag.py --dataset chemical_compound --alpha 0.5 --exp_name tablerag_0.5_B200_llama --budget 200 --k 200 > logs/tablerag/chemical_0.5_B200_llama.log
# python -u tablerag.py --dataset product --alpha 0.5 --exp_name tablerag_0.5_B200_llama --budget 200 --k 200 > logs/tablerag/product_0.5_B200_llama.log

# python -u tablerag.py --dataset animal --alpha 0.5 --exp_name tablerag_0.5_B500_llama --budget 500 --k 500 > logs/tablerag/animal_0.5_B500_llama.log
# python -u tablerag.py --dataset chemical_compound --alpha 0.5 --exp_name tablerag_0.5_B500_llama --budget 500 --k 500 > logs/tablerag/chemical_0.5_B500_llama.log
# python -u tablerag.py --dataset product --alpha 0.5 --exp_name tablerag_0.5_B500_llama --budget 500 --k 500 > logs/tablerag/product_0.5_B500_llama.log
python -u main.py --dataset animal --index_combine_method weighted --exp_name our_qwen_B500 --budget 500 --k 500 > logs/our_qwen/animal_B500.log
python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name our_qwen_B500 --budget 500 --k 500 > logs/our_qwen/chemical_B500.log
python -u main.py --dataset product --index_combine_method weighted --exp_name our_qwen_B500 --budget 500 --k 500 > logs/our_qwen/product_B500.log


python -u main.py --dataset animal --index_combine_method weighted --exp_name our_qwen_B100 --budget 100 --k 100 > logs/our_qwen/animal_B100.log
python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name our_qwen_B100 --budget 100 --k 100 > logs/our_qwen/chemical_B100.log
python -u main.py --dataset product --index_combine_method weighted --exp_name our_qwen_B100 --budget 100 --k 100 > logs/our_qwen/product_B100.log


python -u main.py --dataset animal --index_combine_method weighted --exp_name our_qwen_B200 --budget 200 --k 200 > logs/our_qwen/animal_B200.log
python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name our_qwen_B200 --budget 200 --k 200 > logs/our_qwen/chemical_B200.log
python -u main.py --dataset product --index_combine_method weighted --exp_name our_qwen_B200 --budget 200 --k 200 > logs/our_qwen/product_B200.log