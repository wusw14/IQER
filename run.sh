# python -u main.py --dataset chemical_compound --index_combine_method merge --exp_name our_0623 > logs/our_0623/chemical_merge.log
# python -u main.py --dataset animal --index_combine_method merge --exp_name our_0623 > logs/our_0623/animal_merge.log
python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name our_0623 > logs/our_0623/chemical_weighted.log
python -u main.py --dataset animal --index_combine_method weighted --exp_name our_0623 > logs/our_0623/animal_weighted.log
# python -u main.py --dataset product --index_combine_method merge --exp_name our_0623 > logs/our_0623/product_merge.log
python -u main.py --dataset product --index_combine_method weighted --exp_name our_0623 > logs/our_0623/product_weighted.log
# python -u pneuma.py --dataset animal --exp_name pneuma --alpha 0
# python -u pneuma.py --dataset animal --exp_name pneuma --alpha 1
# python -u pneuma.py --dataset animal --exp_name pneuma --alpha 0.5