import logging

import numpy as np
import pandas as pd
from datasets import Dataset
from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)

logger = logging.getLogger(__name__)
seed = 1


class SADataset():
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()

        # raw_datasets = pd.read_csv("/home/lujia/Datasets/sentiment_data/{}/train.tsv".format(data_args.dataset_name),
        #                            sep="\t")
        self.tokenizer = tokenizer
        self.data_args = data_args
        # labels
        self.num_labels = 2

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        # self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        def dataset_preprocess(raw_datasets):
            return raw_datasets.map(
                self.preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",)
        if data_args.dataset_name in ['twitter', 'offenseval', 'jigsaw']:
            data_dir = 'toxic_data'
        elif data_args.dataset_name in ['amazon', 'yelp', 'imdb']:
            data_dir = 'sentiment_data'
        elif data_args.dataset_name in ['enron']:
            data_dir = 'spam_data'
        else:
            raise ValueError("You have to specify a dataset in the list.")
        if training_args.do_train:
            df_train = pd.read_csv(f"Datasets/{data_dir}/{data_args.dataset_name}/train.tsv", sep="\t")
            raw_train, raw_validate = np.split(df_train.sample(6250, random_state=42), [5000,])
            raw_train = Dataset.from_pandas(raw_train)
            raw_validate = Dataset.from_pandas(raw_validate)
            self.train_dataset = dataset_preprocess(raw_train)
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            if not training_args.do_train:
                df_train = pd.read_csv(f"Datasets/{data_dir}/{data_args.dataset_name}/train.tsv",
                                       sep="\t")
                raw_train, raw_validate = np.split(df_train.sample(6250, random_state=42), [int(.8 * len(df_train))])
                raw_validate = Dataset.from_pandas(raw_validate)
            self.eval_dataset = dataset_preprocess(raw_validate)
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            df_dev = pd.read_csv(f"Datasets/{data_dir}/{data_args.dataset_name}/dev.tsv", sep="\t")
            raw_test = df_dev.sample(1000, random_state=42)
            raw_test = Dataset.from_pandas(raw_test)
            self.predict_dataset = dataset_preprocess(raw_test)
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        # self.metric = load_metric("glue", data_args.dataset_name)

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def compute_metrics(self, p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        # preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        if self.data_args.dataset_name is not None:
            #     result = self.metric.compute(predictions=preds, references=p.label_ids)
            #     if len(result) > 1:
            #         result["combined_score"] = np.mean(list(result.values())).item()
            #     return result
            # elif self.is_regression:
            #     return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            # else:
            return {"accuracy": (preds.argmax(1) == p.label_ids).astype(np.float32).mean().item()}

    def preprocess_function(self, examples):
        # Tokenize the texts
        args = (
            (examples["sentence"],)
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        return result
