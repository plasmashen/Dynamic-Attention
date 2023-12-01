# Dynamic Attention

## Code
### Model Training
```bash
python3 run.py --model_name_or_path bert-large-cased --task_name sa --dataset_name amazon --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 1e-5 --num_train_epochs 10 --pre_seq_len 16 --output_dir checkpoints/amazon-bert-test/ --overwrite_output_dir --hidden_dropout_prob 0.1 --seed 44 --save_strategy no --evaluation_strategy epoch
```
- `--output_dir`: the saving dir for the fine-tuned/prefix-tuned/prompt-tuned model

The model parameter will be saved in `checkpoints` folder

Prompt-tuning model requires larger learning rate.\
Fine-tuning model requires smaller learning rate.


## Attack and Evaluation
Refer to the [README](attack/README.md) in ```attack``` directory
