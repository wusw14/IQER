# python -u main.py --dataset animal --index_combine_method weighted --exp_name our_v4_0628 > logs/our_0628/animal.log
# python -u main.py --dataset product --index_combine_method weighted --exp_name our_v4_0628 > logs/our_0628/product.log
# python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name our_v4_0628 > logs/our_0628/chemical.log
# python -u pneuma.py --dataset animal --exp_name pneuma --alpha 0
# python -u pneuma.py --dataset animal --exp_name pneuma --alpha 1
# python -u pneuma.py --dataset animal --exp_name pneuma --alpha 0.5
# python -u eval_lotus.py --dataset animal --exp_name lotus
# python -u eval_lotus.py --dataset product --exp_name lotus_proxy
# python -u eval_lotus.py --dataset animal --exp_name lotus_proxy
# nohup python -u eval_lotus.py --dataset animal --exp_name debug > logs/lotus/animal.log 2>&1 &
# nohup python -u eval_lotus.py --dataset product --exp_name debug > logs/lotus/product.log 2>&1 &
# nohup python -u eval_lotus.py --dataset chemical_compound --exp_name debug > logs/lotus/chemical.log 2>&1 &

# python -u main.py --dataset animal --index_combine_method weighted --exp_name ab_reform --steps 10 --rethink > logs/ab/animal_reform.log
# python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name ab_reform --steps 10 --rethink > logs/ab/chemical_reform.log
# python -u main.py --dataset product --index_combine_method weighted --exp_name ab_wo_tbval --steps 10 --rethink > logs/ab/product_wo_tbval.log

# python -u main.py --dataset animal --index_combine_method weighted --exp_name tablerag_0.5_B100 --budget 100 --k 100 > logs/tablerag/animal_0.5_B100.log
# python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name tablerag_0.5_B100 --budget 100 --k 100 > logs/tablerag/chemical_0.5_B100.log
# python -u main.py --dataset product --index_combine_method weighted --exp_name tablerag_0.5_B100 --budget 100 --k 100 > logs/tablerag/product_0.5_B100.log

# python -u main.py --dataset animal --index_combine_method weighted --exp_name tablerag_0.5_B200 --budget 200 --k 200 > logs/tablerag/animal_0.5_B200.log
# python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name tablerag_0.5_B200 --budget 200 --k 200 > logs/tablerag/chemical_0.5_B200.log
# python -u main.py --dataset product --index_combine_method weighted --exp_name tablerag_0.5_B200 --budget 200 --k 200 > logs/tablerag/product_0.5_B200.log


python -u main.py --dataset animal --index_combine_method weighted --exp_name our_logistic_v2_B100 --budget 100 --k 100 > logs/our_logistic_v2/animal_B100.log
python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name our_logistic_v2_B100 --budget 100 --k 100 > logs/our_logistic_v2/chemical_B100.log
python -u main.py --dataset product --index_combine_method weighted --exp_name our_logistic_v2_B100 --budget 100 --k 100 > logs/our_logistic_v2/product_B100.log


python -u main.py --dataset animal --index_combine_method weighted --exp_name our_logistic_v2_B200 --budget 200 --k 200 > logs/our_logistic_v2/animal_B200.log
python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name our_logistic_v2_B200 --budget 200 --k 200 > logs/our_logistic_v2/chemical_B200.log
python -u main.py --dataset product --index_combine_method weighted --exp_name our_logistic_v2_B200 --budget 200 --k 200 > logs/our_logistic_v2/product_B200.log

python -u main.py --dataset animal --index_combine_method weighted --exp_name our_logistic_v2_B500 --budget 500 --k 500 > logs/our_logistic_v2/animal_B500.log
python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name our_logistic_v2_B500 --budget 500 --k 500 > logs/our_logistic_v2/chemical_B500.log
python -u main.py --dataset product --index_combine_method weighted --exp_name our_logistic_v2_B500 --budget 500 --k 500 > logs/our_logistic_v2/product_B500.log
