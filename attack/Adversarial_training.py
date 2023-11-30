# import os
import sys
sys.path.append("..")
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import textattack
import transformers
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from models import BertForSequenceClassification, BertPrefixForSequenceClassification, \
    BertPromptForSequenceClassification
import argparse
# from model.sequence_classification import (
#     RobertaPrefixForSequenceClassification,
#     RobertaPromptForSequenceClassification,
#     DebertaPrefixForSequenceClassification
# )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='normal', type=str, choices=["normal", "prefix", "prompt"])
    parser.add_argument("--attack", default='a2t', type=str, choices=["textbugger", "textfooler", "pwws", "a2t"])
    parser.add_argument("--model_dir", type=str)
    # parser.add_argument("--attack_method", type=str, required=True, choices=["textbugger", "textfooler", "pwws"])
    args = parser.parse_args()
    # model_name = 'amazon-bert-' + args.model_name
    tokenizer = AutoTokenizer.from_pretrained('bert-large-cased', model_max_length=256)
    config = AutoConfig.from_pretrained('bert-large-cased')
    if args.model_name=='prefix' or 'prefix' in args.model_dir:
        config.hidden_dropout_prob = 0.1
        config.pre_seq_len = 16
        config.prefix_projection = False
        config.prefix_hidden_size = 512
        lr = 1e-3
        epoch = 13
        clean_epoch = 10
        if args.model_dir:
            epoch = 3
            clean_epoch = 0
            print("Prefix model, clean epoch: 0, total epoch:3")
            model = BertPrefixForSequenceClassification.from_pretrained('../checkpoints/{}'.format(args.model_dir), config=config)
        else:
            epoch = 13
            clean_epoch = 10
            model = BertPrefixForSequenceClassification.from_pretrained("bert-large-cased", config=config)
    elif args.model_name=='prompt' or 'prompt' in args.model_dir:
        config.pre_seq_len = 16
        lr = 1e-2
        if args.model_dir:
            epoch = 3
            clean_epoch = 0
            print("Prompt model, clean epoch: 0, total epoch:3")
            model = BertPromptForSequenceClassification.from_pretrained('../checkpoints/{}'.format(args.model_dir), config=config)
        else:
            epoch = 13
            clean_epoch = 10
            model = BertPromptForSequenceClassification.from_pretrained("bert-large-cased", config=config)
    else:
        lr = 2e-5
        if args.model_dir:
            epoch = 3
            clean_epoch = 0
            model = AutoModelForSequenceClassification.from_pretrained('../checkpoints/{}'.format(args.model_dir), config=config)
        else:
            epoch = 6
            clean_epoch = 3
            model = AutoModelForSequenceClassification.from_pretrained("bert-large-cased", config=config)

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    if args.attack == 'a2t':
        attack = textattack.attack_recipes.A2TYoo2021.build(model_wrapper)
    elif args.attack == 'textbugger':
        attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)
    elif args.attack == 'textfooler':
        attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    elif args.attack == 'pwws':
        attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)

    df_train = pd.read_csv("../../Datasets/sentiment_data/amazon/train.tsv", sep="\t")
    raw_train, raw_validate = np.split(df_train.sample(6250, random_state=42), [5000,])
    # raw_train, raw_validate = np.split(df_train.sample(1000, random_state=42), [800, ])
    # df_train = df_train.sample(50000, random_state=2020)
    sentences_train = raw_train.sentence.tolist()
    labels_train = raw_train.label.values.tolist()
    sentences_val = raw_validate.sentence.tolist()
    labels_val = raw_validate.label.values.tolist()

    train_dataset = [i for i in zip(sentences_train, labels_train)]
    train_dataset = textattack.datasets.Dataset(train_dataset)

    eval_dataset = [i for i in zip(sentences_val, labels_val)]
    eval_dataset = textattack.datasets.Dataset(eval_dataset)

    # Train for 3 epochs with 1 initial clean epochs, 1000 adversarial examples per epoch, learning rate of 5e-5,
    # and effective batch size of 32 (8x4).

    training_args = textattack.TrainingArgs(
        num_epochs=epoch,
        num_clean_epochs=clean_epoch,
        num_train_adv_examples=1000,
        learning_rate=lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        log_to_tb=True,
    )
    # print(training_args)
    trainer = textattack.Trainer(
        model_wrapper,
        "classification",
        attack,
        train_dataset,
        eval_dataset,
        training_args,
    )
    trainer.train()
