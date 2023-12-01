CUDA_VISIBLE_DEVICES=5 python3 TextAttacker.py --model_name amazon-bert-normal --attack_method textbugger --random_top --random_bound 0.1 0.2
CUDA_VISIBLE_DEVICES=5 python3 TextAttacker.py --model_name amazon-bert-normal --attack_method textbugger --random_top --random_bound 0.1 0.2 --dropout

CUDA_VISIBLE_DEVICES=5 python3 TextAttacker.py --model_name amazon-bert-normal --attack_method textfooler --random_top --random_bound 0.1 0.2
CUDA_VISIBLE_DEVICES=5 python3 TextAttacker.py --model_name amazon-bert-normal --attack_method textfooler --random_top --random_bound 0.1 0.2 --dropout

CUDA_VISIBLE_DEVICES=5 python3 TextAttacker.py --model_name amazon-bert-normal --attack_method pwws --random_top --random_bound 0.1 0.2
CUDA_VISIBLE_DEVICES=5 python3 TextAttacker.py --model_name amazon-bert-normal --attack_method pwws --random_top --random_bound 0.1 0.2 --dropout
