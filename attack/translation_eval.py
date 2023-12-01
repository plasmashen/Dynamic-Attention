import html, torch
import nltk
import numpy as np
import tqdm
from modeling_t5 import T5ForTextToText
from textattack.models.tokenizers import T5Tokenizer

print("Reading adversarial text")
f = open('adv_nmt/t5-en-fr/textfooler.txt', encoding='utf-8')
txt = f.read()
text = txt.split('--------------------------------------------- Result ')
orig_sent_, adv_sent_, all_sent_, all_trans, orig_trans, adv_trans = [], [], [], [], [], []
for i in range(1, int(text[-1].split(" ")[0])):
    tmp0 = text[i].split('\n')
    all_sent_.append(tmp0[3])
    #     all_label.append(tmp0[1][2])
    if 'FAILED' in tmp0[1]:
        all_trans.append(tmp0[1].split(' --> ')[0])
    elif 'SKIPPED' in tmp0[1]:
        all_trans.append(tmp0[1].split(' --> ')[0])
    else:
        all_trans.append(tmp0[1].split(' --> ')[0])
        orig_trans.append(tmp0[1].split(' --> ')[0])
        adv_trans.append(tmp0[1].split(' --> ')[1])
        orig_sent_.append(tmp0[3])
        adv_sent_.append(tmp0[5])
all_sent = [i.replace('[', '').replace(']', '') for i in all_sent_]
orig_sent = [i.replace('[', '').replace(']', '') for i in orig_sent_]
adv_sent = [i.replace('[', '').replace(']', '') for i in adv_sent_]

print("Loading model")
model = T5ForTextToText.from_pretrained("t5-en-fr")
tokenizer = T5Tokenizer("english_to_french", max_length=200)
model.output_max_length = 100
device = torch.device("cuda:1")
model.to(device)
model.eval()

print("Predicting original translation")
orig_preds = []
for i in tqdm.tqdm(range(len(orig_trans))):
    orig_ids = tokenizer([orig_sent[i]], padding=True)
    orig_out = model(torch.tensor(orig_ids["input_ids"]).to(device))
    orig_out0 = orig_out[0].replace('<pad> ', '').replace('<unk>', '').replace('</s>', '')
    orig_preds.append(orig_out0)

print("Calculate original adversarial BLEU")
model.model.encoder.decay_value, model.model.encoder.dynamic_attention = 1, False
model.model.decoder.decay_value, model.model.decoder.dynamic_attention = 1, False
bleu_values = []
for i in tqdm.tqdm(range(len(orig_trans))):
    adv_ids = tokenizer([adv_sent[i]], padding=True)
    adv_out = model(torch.tensor(adv_ids["input_ids"]).to(device))
    adv_out0 = adv_out[0].replace('<pad> ', '').replace('<unk>', '').replace('</s>', '')
    bleu_values.append(nltk.translate.bleu_score.sentence_bleu([adv_out0.split(' ')],
                                                               orig_preds[i].split(' ')))
print(np.mean(bleu_values))


# model.eval()
print("Evaluating hyperparameters")
model.model.encoder.train();
model.model.decoder.eval();
dv = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rbs = [[0.1, 0.2, 0.], [0.2, 0.3, 0.], [0.1, 0.3, 0.], [0.2, 0.3, 0.1], [0.3, 0.4, 0.1], [0.2, 0.4, 0.1],
       [0.3, 0.4, 0.2], [0.4, 0.5, 0.2], [0.3, 0.5, 0.2]]
for j in dv:
    model.model.encoder.decay_value, model.model.encoder.dynamic_attention = j, True
    model.model.decoder.decay_value, model.model.decoder.dynamic_attention = 1, False
    bleu_values = []
    for rb in rbs:
        for i in tqdm.tqdm(range(len(orig_trans))):
            adv_ids = tokenizer([adv_sent[i]], padding=True)
            text_length = len(adv_ids["input_ids"][0])
            top1, top2, top3 = int(rb[0] * text_length), int(rb[1] * text_length), int(rb[2] * text_length)
            model.model.encoder.da_range = [top1, top2, top3]

            adv_out = model(torch.tensor(adv_ids["input_ids"]).to(device))
            adv_out0 = adv_out[0].replace('<pad> ', '').replace('<unk>', '').replace('</s>', '')
            bleu_values.append(nltk.translate.bleu_score.sentence_bleu([adv_out0.split(' ')],
                                                                       orig_preds[i].split(' ')))
        print(j, rb, np.mean(bleu_values))
