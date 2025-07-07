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

python -u main.py --dataset animal --index_combine_method weighted --exp_name our_v5 --early_stop > logs/our/animal_v5.log
python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name our_v5 --early_stop > logs/our/chemical_v5.log
python -u main.py --dataset product --index_combine_method weighted --exp_name our_v5 --early_stop > logs/our/product_v5.log