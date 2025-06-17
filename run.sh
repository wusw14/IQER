# python -u main.py --dataset chemical_compound --index_combine_method merge --exp_name our_0617 > logs/our_0617/chemical_merge.log
# python -u main.py --dataset animal --index_combine_method merge --exp_name our_0617 > logs/our_0617/animal_merge.log
# python -u main.py --dataset chemical_compound --index_combine_method weighted --exp_name our_0617 > logs/our_0617/chemical_weighted.log
# python -u main.py --dataset animal --index_combine_method weighted --exp_name our_0617 > logs/our_0617/animal_weighted.log
# python -u main.py --dataset product --index_combine_method merge --exp_name our_0617 > logs/our_0617/product_merge.log
python -u main.py --dataset product --index_combine_method weighted --exp_name our_0617 > logs/our_0617/product_weighted.log

