import torch, tqdm
from torch.nn import CrossEntropyLoss, CosineSimilarity
from models import BertForSequenceClassification, BertPrefixForSequenceClassification, \
    BertPromptForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

cossim = CosineSimilarity(dim=0, eps=1e-6)

config = AutoConfig.from_pretrained('bert-large-cased')
config.hidden_dropout_prob = 0.1
config.pre_seq_len = 16
config.prefix_projection = False
config.prefix_hidden_size = 512
model = BertForSequenceClassification.from_pretrained('../checkpoints/amazon-bert-normal', epsilon=0.)
# model = BertPrefixForSequenceClassification.from_pretrained('../checkpoints/amazon-bert-prefix', config=config, epsilon=0.)
# model = BertPromptForSequenceClassification.from_pretrained('../checkpoints/amazon-bert-prompt', config=config, epsilon=0.)

tokenizer = AutoTokenizer.from_pretrained('bert-large-cased', model_max_length=256)

device = torch.device("cuda:4")
model.to(device);


def sentences_pred(sents, cpu=False, top=[], decay_value=0, random_top=False, random_bound=[3, 6], dropout=False):
    encoded_dict = tokenizer(sents, add_special_tokens=True, max_length=256, padding='max_length',
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


def embeds_pred(embed, mask, cpu=False, random_top=True, random_bound=[3, 6]):
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(inputs_embeds=embed.to(device), attention_mask=mask.to(device),
                       output_attentions=True, output_hidden_states=False, random_top=True, random_bound=[3, 6])
    if cpu:
        logits = logits.logits.cpu()
    return logits


list_of_files = ['topnocl.txt', ]
mypath = "/home/lujia/Dynamic-Attention/attack/adv_output/amazon-bert-normal/pwws"
all_ASR = []
samples = 1
for k, fname in enumerate(list_of_files):
    f = open(mypath + '/' + fname)
    txt = f.read()
    text = txt.split('--------------------------------------------- Result ')
    orig_sent_, adv_sent_, all_sent_, label, all_label = [], [], [], [], []
    for i in range(1, 501):
        tmp0 = text[i].split('\n')
        all_sent_.append(tmp0[3])
        #     all_label.append(tmp0[1][2])
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
    rb, dv = [5, 20], 0.1
    # top = [0.5]
    # rbs = [[1,20],[3, 20]]
    # dvs = 0, 0.1, 0.2, 0.3, 0.4, 0.5
    # for rb in rbs:
    #     for dv in dvs:
    avg1, avg2 = 0, 0
    for _ in range(times):
        count = 0
        for i, j in tqdm.tqdm(zip(all_sent, all_label)):
            preds = []
            for _ in range(1):
                pred = sentences_pred(i, True, random_top=False, random_bound=rb, dropout=False, decay_value=dv)  # disabling dropout
                preds.append(pred)
            if sum(torch.cat(preds).argmax(1) == int(j)) < samples:
                count += 1
        avg1 += count / len(all_label)
    for _ in range(times):
        count = 0
        for i, j in tqdm.tqdm(zip(adv_sent, label)):
            preds = []
            for _ in range(1):
                pred = sentences_pred(i, True, random_top=False, random_bound=rb, dropout=False, decay_value=dv)  # disabling dropout
                preds.append(pred)
            if sum(torch.cat(preds).argmax(1) == int(j)) < samples:
                count += 1
        avg2 += count / len(adv_sent)
    print('random bound: {}, decay value: {}, ACC: {:.2f}%, ASR: {:.2f}%'.format(rb, dv, (1-avg1 / times)*100, avg2*100/times))