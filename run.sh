# python -u main.py --dataset chemical_compound --index_combine_method merge --exp_name ab_norm_rank > logs/ab_norm_rank/chemical_merge.log
# python -u main.py --dataset animal --index_combine_method merge --exp_name ab_norm_rank > logs/ab_norm_rank/animal_merge.log
python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name ab_norm_rank > logs/ab_norm_rank/chemical.log
python -u main.py --dataset animal --index_combine_method weighted --exp_name ab_norm_rank > logs/ab_norm_rank/animal.log
# python -u main.py --dataset product --index_combine_method weighted --exp_name ab_norm_rank > logs/ab_norm_rank/product.log
# python -u pneuma.py --dataset animal --exp_name pneuma --alpha 0
# python -u pneuma.py --dataset animal --exp_name pneuma --alpha 1
# python -u pneuma.py --dataset animal --exp_name pneuma --alpha 0.5