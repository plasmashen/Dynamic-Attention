
# Dynamic-Attention
Created from P-Tuning-v2

## Code

### Adversarial Training 
```bash
python3 Adversarial_training.py --model_name [A]
```
- \[A]: prefix/prompt/None

```bash
python3 Adversarial_training.py --model_dir [A]
```
- \[A]: amazon-bert-normal/amazon-bert-prefix/amazon-bert-prompt

### Adversarial Attack
```bash
python3 TextAttacker.py --attack_method [A] --max_rate [B] --model_name [C]
```
- \[A]: textbugger, textfooler, pwws, bae, deepwordbug, pruthi, checklist, bert\_attack, a2t
- \[B]: usually 0.1 and 0.2.
- \[C]: amazon-bert-normal, amazon-bert-prefix, amazon-bert-prompt
- extra arguments:
  - `--model_dir`: load model from `output` directory where the model is adversarialy trained (str)
  - `--tms`: target max score (default 0.4)
  - `--long_text`: to constrain the range of $m$, should be enabled for Enron dataset
  - `--random_top`: randomize the model attention masking (store_true)
  - `--random_bound`: control the randomized attention masking range (list)
  - `--dropout`: set model to train mode where dropout is on (store_true)

### Inference
Inference examples can be found in `test_top.py` and `test_nmt.py`.
Change the model dir and adversarial example dir.

### Adaptive Attack 
```bash
python3 AdaptiveAttacker.py --model_name amazon-bert-normal --attack_method textbugger --random_top --adaptive 2
```


## Issues
```bash
tensorflow.python.framework.errors_impl.UnknownError: Graph execution error:
JIT compilation failed.
```
Downgrade tensorflow to 2.8.0
