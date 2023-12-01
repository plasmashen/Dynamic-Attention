
# Dynamic-Attention
Created from P-Tuning-v2

## Code

### Adversarial Training 
Train from scratch:
```bash
python3 Adversarial_training.py --model_name [A]
```
- \[A]: prefix/prompt/None

Train from tuned model:
```bash
python3 Adversarial_training.py --model_dir [A]
```
Provide the dir of the model parameter in `checkpoints```
- \[A]: amazon-bert-normal/amazon-bert-prefix/amazon-bert-prompt

### Adversarial Attack

Text classification tasks
```bash
python3 TextAttacker.py --attack_method [A] --model_name [B]
```
- \[A]: textbugger, textfooler, pwws, bae, deepwordbug, pruthi, checklist, bert\_attack, a2t
- \[B]: directory name of model parameters in `checkpoints` 
- extra arguments:
  - `--model_dir`: load model from `output` directory where the model is adversarialy trained (str)
  - `--long_text`: to constrain the upper bound of $m$'s the range, should be enabled for Enron dataset
  - `--dynamic_attention`: enable the dynamic attention masking (store_true)
  - `--da_range`: control the range of $m$, usually 0.1 and 0.2 for fine-tuned model (list)
  - `--dropout`: enable dropout (store_true)

The adversarial example txt file will be saved in `adv_output` folder. 

Text generation tasks
```bash
python3 NMTAttacker.py --attack_method [A] --model_name [B]
```


### Inference
Inference examples can be found in `test_top.py` and `test_nmt.py`.
Change the model dir and adversarial example dir.

### Adaptive Attack 
```bash
python3 AdaptiveAttacker.py --model_name amazon-bert-normal --attack_method textbugger --random_top --adaptive [A]
```
- \[A]: the adaptive attack type stated in paper (1 or 2)


## Issues
```bash
tensorflow.python.framework.errors_impl.UnknownError: Graph execution error:
JIT compilation failed.
```
Downgrade tensorflow to 2.8.0
