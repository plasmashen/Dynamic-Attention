#CUDA_VISIBLE_DEVICES=2 python3 TextAttacker.py --model_name amazon-gpt2 --attack_method textbugger --dropout
#CUDA_VISIBLE_DEVICES=2 python3 TextAttacker.py --model_name amazon-gpt2 --attack_method textfooler --dropout
#CUDA_VISIBLE_DEVICES=2 python3 TextAttacker.py --model_name amazon-gpt2 --attack_method pwws --dropout

CUDA_VISIBLE_DEVICES=2 python3 TextAttacker.py --model_name amazon-gpt2 --attack_method textbugger --random_top --random_bound 0.1 0.9 --decay_value 0.4
CUDA_VISIBLE_DEVICES=2 python3 TextAttacker.py --model_name amazon-gpt2 --attack_method textbugger --random_top --random_bound 0.1 0.9 --dropout --decay_value 0.4
CUDA_VISIBLE_DEVICES=2 python3 TextAttacker.py --model_name amazon-gpt2 --attack_method textbugger --random_top --random_bound 0. 0.99 --decay_value 0.4
CUDA_VISIBLE_DEVICES=2 python3 TextAttacker.py --model_name amazon-gpt2 --attack_method textbugger --random_top --random_bound 0. 0.99 --dropout --decay_value 0.4
#CUDA_VISIBLE_DEVICES=3 python3 TextAttacker.py --model_name amazon-gpt2 --attack_method textfooler --random_top --random_bound 0.1 0.2
#CUDA_VISIBLE_DEVICES=3 python3 TextAttacker.py --model_name amazon-gpt2 --attack_method textfooler --random_top --random_bound 0.1 0.2 --dropout
#CUDA_VISIBLE_DEVICES=3 python3 TextAttacker.py --model_name amazon-gpt2 --attack_method pwws --random_top --random_bound 0.1 0.2
#CUDA_VISIBLE_DEVICES=3 python3 TextAttacker.py --model_name amazon-gpt2 --attack_method pwws --random_top --random_bound 0.1 0.2 --dropout
