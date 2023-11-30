from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from os import listdir
from os.path import isfile, join
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import BertForSequenceClassification, BertPrefixForSequenceClassification, \
    BertPromptForSequenceClassification

device = torch.device('cuda:1')

# device = torch.device('cpu')

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default='normal', type=str, choices=["normal", "prefix", "prompt"])
parser.add_argument("--attack_method", type=str, required=True, choices=["textbugger", "textfooler", "pwws"])
# parser.add_argument("--data_form", type=str, required=True, choices=["clean", "adv"])
parser.add_argument("--top", nargs='+', type=int, default=[])
# parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)

args = parser.parse_args()
model_name = 'amazon-bert-'+args.model_name
attack_method = args.attack_method
# data_form = args.data_form
top = [int(i) for i in args.top]


def sentences_pred(sents, batch_size=1):
    features = []
    probs_all = []
    for sent in sents:
        encoded_dict = tokenizer.encode_plus(sent, add_special_tokens=True, max_length=256,
                                             padding='max_length', return_attention_mask=True,
                                             return_tensors='pt', truncation=True)
        features.append(encoded_dict)
    iids = torch.cat([f['input_ids'] for f in features])
    amasks = torch.cat([f['attention_mask'] for f in features])
    eval_data = TensorDataset(iids, amasks)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size)
    bert.to(device)
    for input_ids, input_mask in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        with torch.no_grad():
            logits = bert(input_ids, token_type_ids=None, attention_mask=input_mask, output_attentions=True, top=top)[0]
            probs = nn.functional.softmax(logits, dim=-1)
            probs_all.append(probs)

    return torch.cat(probs_all, dim=0)


if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained('../checkpoints/amazon-bert-normal', model_max_length=256)
    tokenizer = AutoTokenizer.from_pretrained('../checkpoints/{}'.format(model_name), model_max_length=256)
    if 'prefix' in model_name:
        bert = BertPrefixForSequenceClassification.from_pretrained('../checkpoints/{}'.format(model_name), output_attentions=True)
    elif 'prompt' in model_name:
        bert = BertPromptForSequenceClassification.from_pretrained('../checkpoints/{}'.format(model_name))
    else:
        bert = BertForSequenceClassification.from_pretrained('../checkpoints/{}'.format(model_name))
    mypath = 'adv_output/{}/{}'.format(model_name, attack_method)
    onlyfiles = sorted([f for f in listdir(mypath) if isfile(join(mypath, f))])
    f = open(mypath + '/{}'.format(onlyfiles[-1]))
    txt = f.read()
    text = txt.split('--------------------------------------------- Result ')
    orig_sent_, adv_sent_, label = [], [], []
    count = 0
    for i in range(1, 501):
        tmp0 = text[i].split('\n')
        if 'FAILED' in tmp0[1] or 'SKIPPED' in tmp0[1]:
            pass
        else:
            label.append(tmp0[1][2])
            orig_sent_.append(tmp0[3])
            adv_sent_.append(tmp0[5])
    orig_sent = [i.replace('[', '').replace(']', '') for i in orig_sent_]
    adv_sent = [i.replace('[', '').replace(']', '') for i in adv_sent_]
    labeli = [int(i) for i in label]
    adv_pred = sentences_pred(adv_sent).cpu()
    orig_pred = sentences_pred(orig_sent).cpu()
    adv_acc = sum([i == j for i, j in zip(adv_pred.argmax(1), labeli)]) / len(labeli)
    orig_acc = sum([i == j for i, j in zip(orig_pred.argmax(1), labeli)]) / len(labeli)
    print('adv_acc: ', adv_acc, '\norig_acc: ', orig_acc)
# -10000*((torch.rand_like(attention_mask)*(attention_mask==0))>0.8)+attention_mask
