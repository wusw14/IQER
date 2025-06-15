# python -u main.py --dataset plant --output_dir results/debug --method llm_check > logs/plant.log
# python -u main.py --dataset animal --output_dir results/debug --method llm_check > logs/animal.log
# python -u main.py --dataset paper --output_dir results/debug --method llm_check > logs/paper.log
# python -u main.py --dataset product --output_dir results/debug --method llm_check > logs/product.log
# python -u main.py --dataset chemical_compound --output_dir results/debug --method llm_check > logs/chemical.log
python -u main.py --dataset transportation_device --output_dir results/debug --method llm_check > logs/trans.log

python -u index.py plant
python -u index.py animal
python -u index.py paper
python -u index.py product
python -u index.py chemical_compound
python -u index.py transportation_device


python -u test_llm.py --dataset plant --output_dir results/llm --method llm_check > logs/plant.log
python -u test_llm.py --dataset chemical_compound --output_dir results/llm --method llm_check > logs/chemical.log
python -u test_llm.py --dataset transportation_device --output_dir results/llm --method llm_check > logs/trans.log
python -u test_llm.py --dataset product --output_dir results/llm --method llm_check > logs/product.log
python -u test_llm.py --dataset animal --output_dir results/llm --method llm_check > logs/animal.log
python -u test_llm.py --dataset paper --output_dir results/llm --method llm_check > logs/paper.log

python -u main.py --dataset chemical_compound --output_dir results/our_v1 --method llm_check > logs/our_v1/chemical.log
python -u main.py --dataset plant --output_dir results/our_v1 --method llm_check > logs/our_v1/plant.log
python -u main.py --dataset animal --output_dir results/our_v1 --method llm_check > logs/our_v1/animal.log

python -u main.py --dataset chemical_compound --output_dir results/base --method llm_check > logs/base/chemical.log
python -u main.py --dataset plant --output_dir results/base --method llm_check > logs/base/plant.log
python -u main.py --dataset animal --output_dir results/base --method llm_check > logs/base/animal.log
python -u main.py --dataset paper --output_dir results/base --method llm_check > logs/base/paper.log
python -u main.py --dataset product --output_dir results/base --method llm_check > logs/base/product.log
python -u main.py --dataset transportation_device --output_dir results/base --method llm_check > logs/base/trans.log