import torch
import numpy as np
from models import BertForSequenceClassification, BertPrefixForSequenceClassification, \
    BertPromptForSequenceClassification
from modeling_gpt2 import GPT2ForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

# Use the corresponding model for fine-tuned, prefix-tuned  and prompt tuned model
# model = GPT2ForSequenceClassification.from_pretrained('../checkpoints/amazon-gpt2')
model = BertPromptForSequenceClassification.from_pretrained('../checkpoints/amazon-bert-prompt')
tokenizer = AutoTokenizer.from_pretrained('../checkpoints/amazon-bert-prompt')
device = torch.device("cuda")
model.to(device);


def sentences_pred(sents, cpu=False, top=[], decay_value=0., random_top=False, random_bound=[3, 6], dropout=False):
    encoded_dict = tokenizer(sents, add_special_tokens=True, max_length=128, padding='max_length',
                             return_attention_mask=True, return_tensors='pt', truncation=True)
    model.to(device)
    if dropout:
        model.train()
    else:
        model.eval()
    input_ids = encoded_dict['input_ids'].to(device)
    input_mask = encoded_dict['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, token_type_ids=None, attention_mask=input_mask, decay_value=decay_value,
                       output_attentions=True, output_hidden_states=True, top=top,
                       random_top=random_top, random_bound=random_bound)
    if cpu:
        logits = logits.logits.cpu()
    return logits

f = open('path_to_adversarial_examples.txt')
txt = f.read()
text = txt.split('--------------------------------------------- Result ')
orig_sent_, adv_sent_, all_sent_, label, all_label = [], [], [], [], []
for i in range(1, 501):
    tmp0 = text[i].split('\n')
    all_sent_.append(tmp0[3])
    if 'FAILED' in tmp0[1]:
        all_label.append(tmp0[1][2])
    elif 'SKIPPED' in tmp0[1]:
        if tmp0[1][2] == '1':
            all_label.append('0')
        else:
            all_label.append('1')
    else:
        all_label.append(tmp0[1][2])
        label.append(tmp0[1][2])
        orig_sent_.append(tmp0[3])
        adv_sent_.append(tmp0[5])
all_sent = [i.replace('[', '').replace(']', '') for i in all_sent_]
orig_sent = [i.replace('[', '').replace(']', '') for i in orig_sent_]
adv_sent = [i.replace('[', '').replace(']', '') for i in adv_sent_]


times = 1
avg = []
dv = 0. # \beta
rb = [0., 0.2] # m's range in proportion w.r.t. the text length
for _ in range(times):
    count = 0
    for i, j in zip(all_sent, all_label):
        preds = []
        l = len(i.split(' '))
        top1 = int(rb[0] * l)
        top2 = int(rb[1] * l) # the specific m's range
        for _ in range(1):
            pred = sentences_pred(i, True, decay_value=dv, random_top=True, random_bound=[top1, top2],
                                  dropout=False)
            preds.append(pred)
        if torch.cat(preds).argmax(1) == int(j):
            count += 1
    avg.append(count / len(all_sent))
print("adversarial accuracy {:.4f}".format(np.mean(avg)))

for _ in range(times):
    count = 0
    for i, j in zip(adv_sent, label):
        preds = []
        l = len(i.split(' '))
        top1 = int(rb[0] * l)
        top2 = int(rb[1] * l)
        for _ in range(1):
            pred = sentences_pred(i, True, decay_value=dv, random_top=True, random_bound=[top1, top2],
                                  dropout=False)
            preds.append(pred)
        if torch.cat(preds).argmax(1) == int(j):
            count += 1
    avg.append(count / len(adv_sent))
print("clean accuracy {:.4f}".format(np.mean(avg)))
