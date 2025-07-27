# python -u eval_lotus.py --dataset chemical_compound --exp_name lotus_proxy --llm qwen > logs/lotus_proxy/chemical_qwen.log
# python -u eval_lotus.py --dataset animal --exp_name lotus_proxy --llm qwen > logs/lotus_proxy/animal_qwen.log
# python -u eval_lotus.py --dataset product --exp_name lotus_proxy --llm qwen > logs/lotus_proxy/product_qwen.log

# python -u main.py --dataset animal --index_combine_method weighted --exp_name ab_reform --steps 10 --rethink > logs/ab/animal_reform.log
# python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name ab_reform --steps 10 --rethink > logs/ab/chemical_reform.log
# python -u main.py --dataset product --index_combine_method weighted --exp_name ab_wo_tbval --steps 10 --rethink > logs/ab/product_wo_tbval.log

# python -u tablerag.py --dataset animal --alpha 0.5 --exp_name tablerag_0.5_B100_qwen --budget 500 --k 500 > logs/tablerag/animal_0.5_B100_qwen.log
# python -u tablerag.py --dataset chemical_compound --alpha 0.5 --exp_name tablerag_0.5_B100_qwen --budget 500 --k 500 > logs/tablerag/chemical_0.5_B100_qwen.log
# python -u tablerag.py --dataset product --alpha 0.5 --exp_name tablerag_0.5_B100_qwen --budget 500 --k 500 > logs/tablerag/product_0.5_B100_qwen.log

# python -u tablerag.py --dataset animal --alpha 0.5 --exp_name tablerag_0.5_B200_qwen --budget 200 --k 200 > logs/tablerag/animal_0.5_B200_qwen.log
# python -u tablerag.py --dataset chemical_compound --alpha 0.5 --exp_name tablerag_0.5_B200_qwen --budget 200 --k 200 > logs/tablerag/chemical_0.5_B200_qwen.log
# python -u tablerag.py --dataset product --alpha 0.5 --exp_name tablerag_0.5_B200_qwen --budget 200 --k 200 > logs/tablerag/product_0.5_B200_qwen.log

# python -u tablerag.py --dataset animal --alpha 0.5 --exp_name tablerag_0.5_B500_qwen --budget 500 --k 500 > logs/tablerag/animal_0.5_B500_qwen.log
# python -u tablerag.py --dataset chemical_compound --alpha 0.5 --exp_name tablerag_0.5_B500_qwen --budget 500 --k 500 > logs/tablerag/chemical_0.5_B500_qwen.log
# python -u tablerag.py --dataset product --alpha 0.5 --exp_name tablerag_0.5_B500_qwen --budget 500 --k 500 > logs/tablerag/product_0.5_B500_qwen.log
# python -u main.py --dataset animal --exp_name qwen_early_stop_B100 --budget 500 --k 500 --early_stop > logs/qwen_early_stop/animal_B100.log
# python -u main.py --dataset chemical_compound --exp_name qwen_early_stop_B100 --budget 500 --k 500 --early_stop > logs/qwen_early_stop/chemical_B100.log
# python -u main.py --dataset product --exp_name qwen_early_stop_B100 --budget 500 --k 500 --early_stop > logs/qwen_early_stop/product_B100.log

# python -u main.py --dataset animal --exp_name qwen_early_stop_B200 --budget 200 --k 200 --early_stop > logs/qwen_early_stop/animal_B200.log
# python -u main.py --dataset chemical_compound --exp_name qwen_early_stop_B200 --budget 200 --k 200 --early_stop > logs/qwen_early_stop/chemical_B200.log
# python -u main.py --dataset product --exp_name qwen_early_stop_B200 --budget 200 --k 200 --early_stop > logs/qwen_early_stop/product_B200.log

# python -u main.py --dataset animal --exp_name qwen_rerank_max_B100 --budget 500 --k 500 --rerank max > logs/qwen_rerank/animal_max_B100.log
# python -u main.py --dataset chemical_compound --exp_name qwen_rerank_max_B100 --budget 500 --k 500 --rerank max > logs/qwen_rerank/chemical_max_B100.log
# python -u main.py --dataset product --exp_name qwen_rerank_max_B100 --budget 500 --k 500 --rerank max > logs/qwen_rerank/product_max_B100.log

# python -u main.py --dataset animal --exp_name qwen_rerank_max_B200 --budget 200 --k 200 --rerank max > logs/qwen_rerank/animal_max_B200.log
# python -u main.py --dataset chemical_compound --exp_name qwen_rerank_max_B200 --budget 200 --k 200 --rerank max > logs/qwen_rerank/chemical_max_B200.log
# python -u main.py --dataset product --exp_name qwen_rerank_max_B200 --budget 200 --k 200 --rerank max > logs/qwen_rerank/product_max_B200.log

# python -u main.py --dataset animal --exp_name qwen_0727_B100 --budget 100 --k 100 --rerank search > logs/qwen_0727/animal_search_B100.log
# python -u main.py --dataset chemical_compound --exp_name qwen_0727_B100 --budget 100 --k 100 --rerank search > logs/qwen_0727/chemical_search_B100.log
# python -u main.py --dataset product --exp_name qwen_0727_B100 --budget 100 --k 100 --rerank search > logs/qwen_0727/product_search_B100.log

# python -u main.py --dataset animal --exp_name qwen_0727_B200 --budget 200 --k 200 --rerank search > logs/qwen_0727/animal_search_B200.log
# python -u main.py --dataset chemical_compound --exp_name qwen_0727_B200 --budget 200 --k 200 --rerank search > logs/qwen_0727/chemical_search_B200.log
# python -u main.py --dataset product --exp_name qwen_0727_B200 --budget 200 --k 200 --rerank search > logs/qwen_0727/product_search_B200.log

# python -u main.py --dataset animal --exp_name qwen_0727_B500 --budget 500 --k 500 --rerank search > logs/qwen_0727/animal_search_B500.log
# python -u main.py --dataset chemical_compound --exp_name qwen_0727_B500 --budget 500 --k 500 --rerank search > logs/qwen_0727/chemical_search_B500.log
# python -u main.py --dataset product --exp_name qwen_0727_B500 --budget 500 --k 500 --rerank search > logs/qwen_0727/product_search_B500.log

# python -u main.py --dataset animal --exp_name qwen_ab_stop_tau0.05_alpha1 --budget 500 --k 500 --rerank search --tau 0.05 --alpha 1 --early_stop > logs/qwen_ab_stop/animal_tau0.05_alpha1.log
# python -u main.py --dataset animal --exp_name qwen_ab_stop_tau0.05_alpha2 --budget 500 --k 500 --rerank search --tau 0.05 --alpha 2 --early_stop > logs/qwen_ab_stop/animal_tau0.05_alpha2.log
# python -u main.py --dataset animal --exp_name qwen_ab_stop_tau0.05_alpha3 --budget 500 --k 500 --rerank search --tau 0.05 --alpha 3 --early_stop > logs/qwen_ab_stop/animal_tau0.05_alpha3.log
# python -u main.py --dataset animal --exp_name qwen_ab_stop_tau0.05_alpha4 --budget 500 --k 500 --rerank search --tau 0.05 --alpha 4 --early_stop > logs/qwen_ab_stop/animal_tau0.05_alpha4.log
# python -u main.py --dataset animal --exp_name qwen_ab_stop_tau0.05_alpha5 --budget 500 --k 500 --rerank search --tau 0.05 --alpha 5 --early_stop > logs/qwen_ab_stop/animal_tau0.05_alpha5.log

python -u main.py --dataset animal --exp_name qwen_reform_few_shot_B100 --budget 100 --k 100 --rerank search --reform_type few-shot > logs/qwen_reform/animal_few_shot_B100.log
python -u main.py --dataset animal --exp_name qwen_reform_cot_B100 --budget 100 --k 100 --rerank search --reform_type cot > logs/qwen_reform/animal_cot_B100.log