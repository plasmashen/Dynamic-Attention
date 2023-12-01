import argparse
import sys, re

from textattack import Attack
from textattack import Attacker
from textattack import AttackArgs
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    MaxModificationRate,
    InputColumnModification,
    MinWordLength,
)
from textattack.constraints.overlap import MaxWordsPerturbed, LevenshteinEditDistance
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder, BERT
from textattack.constraints.semantics import WordEmbeddingDistance

from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR, GreedySearch

from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
    WordSwapMaskedLM,
    WordSwapWordNet,
    WordSwapQWERTY,
    WordSwapChangeLocation,
    WordSwapChangeName,
    WordSwapChangeNumber,
    WordSwapContract,
    WordSwapExtend,
)
from textattack.constraints import Constraint
from textattack.attack_recipes import AttackRecipe
from modeling_bert import BertForSequenceClassification, BertPrefixForSequenceClassification, \
    BertPromptForSequenceClassification
# from model.sequence_classification import (
#     BertPrefixForSequenceClassification,
#     BertPromptForSequenceClassification,
#     RobertaPrefixForSequenceClassification,
#     RobertaPromptForSequenceClassification,
#     DebertaPrefixForSequenceClassification
# )
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, random
import numpy as np
import pandas as pd
# from torch.nn import CrossEntropyLoss
import textattack
from ModelWrapper import HuggingFaceModelWrapper

sys.path.append("..")


class Attention1(Constraint):
    def __init__(
            self,
            min_overlap_score,
            model,
            tokenizer,
            compare_against_original=True,
    ):
        super().__init__(compare_against_original)
        if not isinstance(min_overlap_score, float):
            raise TypeError("min_pct_score must be a float")
        if min_overlap_score < 0.0 or min_overlap_score > 1.0:
            raise ValueError("min_pct_score must be a value between 0.0 and 1.0")

        self.min_overlap_score = min_overlap_score
        self.model = model
        self.tokenizer = tokenizer
        # Turn off idf-weighting scheme b/c reference sentence set is small

    def _check_constraint(self, transformed_text, reference_text):
        """Return `True` if high overlap of attentive tokens between `transformed_text` and
        `reference_text`, a threshold is used to determine the constraint"""
        cand = transformed_text.text
        ref = reference_text.text
        cand_result = self.attentive_token_ids(cand)
        ref_result = self.attentive_token_ids(ref)
        pct = len(cand_result.intersection(ref_result)) / len(cand_result.union(ref_result))
        # score = result[BERTScore.SCORE_TYPE2IDX[self.score_type]].item()
        if pct >= self.min_overlap_score:
            return True
        else:
            return False

    def attentive_token_ids(self, text):
        tokenlabel = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        bound_m = len(tokenlabel)
        tokens = tokenizer(text, return_tensors='pt', truncation=True)
        output = self.model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())
        b = []
        for i in range(18, 24):
            b += [tokenlabel[i] for i in
                  output.attentions[i][0, 0, 1:bound_m - 1, 1:bound_m - 1].sum(0).sort().indices[-5:]]
        return set(b)

    def extra_repr_keys(self):
        return ["min_overlap_score", "model", "tokenizer"] + super().extra_repr_keys()


class Attention2(Constraint):
    def __init__(
            self,
            max_std_score,
            model,
            tokenizer,
            compare_against_original=True,
    ):
        super().__init__(compare_against_original)
        if not isinstance(max_std_score, float):
            raise TypeError("max_pct_score must be a float")
        if max_std_score <= 0:
            raise ValueError("max_pct_score must be a value larger than 0")

        self.max_std_score = max_std_score
        self.model = model
        self.tokenizer = tokenizer
        # Turn off idf-weighting scheme b/c reference sentence set is small

    def _check_constraint(self, transformed_text, reference_text):
        """Return `True` if standard deviation of the average attention value of
         `transformed_text` lower than 1.5."""
        cand = transformed_text.text
        # ref = reference_text.text
        cand_result = self.attentive_token_ids(cand)
        # ref_result = self.attentive_token_ids(ref)
        # pct = [i / j for i, j in zip(cand_result, ref_result)]
        # score = result[BERTScore.SCORE_TYPE2IDX[self.score_type]].item()
        if max(cand_result) <= self.max_std_score:
            return True
        else:
            return False

    def attentive_token_ids(self, text):
        tokenlabel = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        bound_m = len(tokenlabel)
        tokens = tokenizer(text, return_tensors='pt', truncation=True)
        output = self.model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())
        b = []
        for i in range(18, 24):
            b.append(np.std(output.attentions[i][0, 0, 1:bound_m - 1, 1:bound_m - 1].sum(0).cpu().detach().numpy()))
        return b

    def extra_repr_keys(self):
        return ["max_std_score", "model", "tokenizer"] + super().extra_repr_keys()


# class Attention3(Constraint):
#     def __init__(
#             self,
#             min_pct_score,
#             model,
#             tokenizer,
#             compare_against_original=True,
#     ):
#         super().__init__(compare_against_original)
#         if not isinstance(min_pct_score, float):
#             raise TypeError("min_pct_score must be a float")
#         if min_pct_score < 0.0 or min_pct_score > 1.0:
#             raise ValueError("min_pct_score must be a value between 0.0 and 1.0")
#
#         self.min_pct_score = min_pct_score
#         self.model = model
#         self.tokenizer = tokenizer
#         # Turn off idf-weighting scheme b/c reference sentence set is small
#
#     def _check_constraint(self, transformed_text, reference_text):
#         """Return `True` if BERT Score between `transformed_text` and
#         `reference_text` is lower than minimum BERT Score."""
#         cand = transformed_text.text
#         ref = reference_text.text
#         cand_result = self.attentive_token_ids(cand)
#         ref_result = self.attentive_token_ids(ref)
#         pct = len(cand_result.intersection(ref_result)) / len(cand_result.union(ref_result))
#         # score = result[BERTScore.SCORE_TYPE2IDX[self.score_type]].item()
#         if pct >= self.min_pct_score:
#             return True
#         else:
#             return False
#
#     def attentive_token_ids(self, text):
#         tokenlabel = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
#         bound_m = len(tokenlabel)
#         tokens = tokenizer(text, return_tensors='pt', truncation=True)
#         output = self.model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())
#         b = []
#         for i in range(18, 24):
#             b += [tokenlabel[i] for i in
#                   output.attentions[i][0, 0, 1:bound_m - 1, 1:bound_m - 1].sum(0).sort().indices[:-10]]
#         return set(b)
#
#     def extra_repr_keys(self):
#         return ["min_pct_score", "model", "tokenizer"] + super().extra_repr_keys()


class BERTAttackLi2020(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate, tms=0.4):
        transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=48)
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(MaxWordsPerturbed(max_percent=0.4))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=tms, query_budget=200)
        search_method = GreedyWordSwapWIR(wir_method="unk")

        return Attack(goal_function, constraints, transformation, search_method)


class TextBuggerLi2018(AttackRecipe):

    @staticmethod
    def build(model_wrapper, max_rate, tms=0.4):
        transformation = CompositeTransformation(
            [
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                WordSwapHomoglyphSwap(),
                WordSwapEmbedding(max_candidates=2),
            ]
        )

        constraints = [RepeatModification(), StopwordModification(), UniversalSentenceEncoder(threshold=0.8),
                       MaxModificationRate(max_rate=max_rate, min_threshold=1)]
        goal_function = UntargetedClassification(model_wrapper, target_max_score=tms)
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)


class TextFoolerJin2019(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate, tms=0.4):
        transformation = WordSwapEmbedding(max_candidates=50)
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost",
             "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any",
             "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at",
             "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between",
             "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't",
             "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty",
             "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former",
             "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here",
             "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however",
             "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just",
             "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more",
             "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't",
             "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing",
             "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others",
             "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't",
             "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere",
             "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence",
             "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those",
             "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until",
             "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever",
             "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
             "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why",
             "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd",
             "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=tms)
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)


class PWWSRen2019(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate, tms=0.4):
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=tms)
        # search over words based on a combination of their saliency score, and how efficient the WordSwap transform is
        search_method = GreedyWordSwapWIR("weighted-saliency")
        return Attack(goal_function, constraints, transformation, search_method)


class Pruthi2019(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate, max_num_word_swaps=1, tms=0.4):
        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterInsertion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapQWERTY(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
            ]
        )
        constraints = [
            MinWordLength(min_length=4),
            StopwordModification(),
            MaxWordsPerturbed(max_num_words=max_num_word_swaps),
            RepeatModification(),
        ]
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=tms, query_budget=200)
        search_method = GreedySearch()
        return Attack(goal_function, constraints, transformation, search_method)


class CheckList2020(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate, tms=0.4):
        transformation = CompositeTransformation(
            [
                WordSwapExtend(),
                WordSwapContract(),
                WordSwapChangeName(),
                WordSwapChangeNumber(),
                WordSwapChangeLocation(),
            ]
        )
        constraints = [RepeatModification()]
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=tms, query_budget=200)
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)


class DeepWordBugGao2018(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate, use_all_transformations=True, tms=0.4):
        if use_all_transformations:
            transformation = CompositeTransformation(
                [
                    WordSwapNeighboringCharacterSwap(),
                    WordSwapRandomCharacterSubstitution(),
                    WordSwapRandomCharacterDeletion(),
                    WordSwapRandomCharacterInsertion(),
                ]
            )
        else:
            transformation = WordSwapRandomCharacterSubstitution()
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(LevenshteinEditDistance(30))
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=tms, query_budget=200)
        search_method = GreedyWordSwapWIR()

        return Attack(goal_function, constraints, transformation, search_method)


class BAEGarg2019(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate, tms=0.4):
        transformation = WordSwapMaskedLM(
            method="bae", max_candidates=50, min_confidence=0.0
        )
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=tms, query_budget=200)
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)


class A2TYoo2021(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate, mlm=False, tms=0.4):
        constraints = [RepeatModification(), StopwordModification()]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        constraints.append(PartOfSpeech(allow_verb_noun_swap=False))
        # constraints.append(MaxModificationRate(max_rate=0.1, min_threshold=4))
        sent_encoder = BERT(
            model_name="stsb-distilbert-base", threshold=0.9, metric="cosine"
        )
        constraints.append(sent_encoder)

        if mlm:
            transformation = transformation = WordSwapMaskedLM(
                method="bae", max_candidates=20, min_confidence=0.0, batch_size=16
            )
        else:
            transformation = WordSwapEmbedding(max_candidates=20)
            constraints.append(WordEmbeddingDistance(min_cos_sim=0.8))
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=4))
        goal_function = UntargetedClassification(model_wrapper, model_batch_size=32, target_max_score=tms,
                                                 query_budget=200)
        search_method = GreedyWordSwapWIR(wir_method="gradient")

        return Attack(goal_function, constraints, transformation, search_method)


def AutoAttack(attack_method):
    if attack_method == "textbugger":
        return TextBuggerLi2018
    if attack_method == "textfooler":
        return TextFoolerJin2019
    if attack_method == "pwws":
        return PWWSRen2019
    if attack_method == "bae":
        return BAEGarg2019
    if attack_method == "deepwordbug":
        return DeepWordBugGao2018
    if attack_method == "pruthi":
        return Pruthi2019
    if attack_method == "checklist":
        return CheckList2020
    if attack_method == "bert_attack":
        return BERTAttackLi2020
    if attack_method == "a2t":
        return A2TYoo2021


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--attack_method", type=str, required=True, choices=["textbugger", "textfooler", "pwws", "bae",
                                                                             "deepwordbug", "pruthi", "checklist",
                                                                             "bert_attack", "a2t"])
    parser.add_argument("--max_rate", default=0.2, type=float)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--decay_value", default=0, type=float)
    parser.add_argument("--tms", default=0.4, type=float)
    parser.add_argument("--dynamic_attention", action="store_true")
    parser.add_argument("--fixed_range", nargs='+', type=int, default=[])
    parser.add_argument("--da_range", nargs='+', type=float, default=[0.1, 0.2])
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--adaptive", default=1, type=int)

    args = parser.parse_args()
    max_rate = args.max_rate
    attack_method = args.attack_method
    model_name = args.model_name
    dataset_name = args.dataset_name
    model_dir = args.model_dir
    decay_value = args.decay_value
    dynamic_attention = args.dynamic_attention
    fixed_range = [int(i) for i in args.fixed_range]
    da_range = [float(i) for i in args.da_range]
    dropout = args.dropout
    target_max_score = args.tms
    adaptive_type = args.adaptive
    dataset_name = dataset_name if dataset_name else model_name

    if model_dir:
        tokenizer = AutoTokenizer.from_pretrained('outputs/{}/last_model'.format(model_dir), model_max_length=256)
        if 'prefix' in model_dir:
            bert = BertPrefixForSequenceClassification.from_pretrained('outputs/{}/last_model'.format(model_dir))
        elif 'prompt' in model_dir:
            bert = BertPromptForSequenceClassification.from_pretrained('outputs/{}/last_model'.format(model_dir))
        else:
            bert = BertForSequenceClassification.from_pretrained('outputs/{}/last_model'.format(model_dir))
    elif model_name:
        if 'prefix' in model_name:
            tokenizer = AutoTokenizer.from_pretrained('../checkpoints/{}'.format(model_name), model_max_length=256)
            bert = BertPrefixForSequenceClassification.from_pretrained('../checkpoints/{}'.format(model_name))
        elif 'prompt' in model_name:
            tokenizer = AutoTokenizer.from_pretrained('../checkpoints/{}'.format(model_name), model_max_length=256)
            bert = BertPromptForSequenceClassification.from_pretrained('../checkpoints/{}'.format(model_name))
        else:
            tokenizer = AutoTokenizer.from_pretrained('../checkpoints/{}'.format(model_name), model_max_length=256)
            bert = BertForSequenceClassification.from_pretrained('../checkpoints/{}'.format(model_name))
    else:
        raise ValueError(f"must provide model_dir or model_name")
    model_wrapper = HuggingFaceModelWrapper(bert, tokenizer, fixed_range, decay_value, dynamic_attention, da_range, dropout)
    attack_module = AutoAttack(attack_method)
    attack = attack_module.build(model_wrapper, max_rate, tms=target_max_score)
    if adaptive_type == 1:
        attack.constraints.append(Attention1(0.8, bert, tokenizer))
    elif adaptive_type == 2:
        attack.constraints.append(Attention2(1.5, bert, tokenizer))
    else:
        raise ValueError(f"adaptive must 1 or 2")
    random.seed(1)

    if 'twitter' in dataset_name:
        df_db_val = pd.read_csv("../Datasets/toxic_data/twitter/dev.tsv", sep="\t")
        dataset_name = 'twitter'
    elif 'jigsaw' in dataset_name:
        df_db_val = pd.read_csv("../Datasets/toxic_data/jigsaw/dev.tsv", sep="\t")
        dataset_name = 'jigsaw'
    elif 'amazon' in dataset_name:
        df_db_val = pd.read_csv("../Datasets/sentiment_data/amazon/dev.tsv", sep="\t")
        dataset_name = 'amazon'
    elif 'yelp' in dataset_name:
        df_db_val = pd.read_csv("../Datasets/sentiment_data/yelp/dev.tsv", sep="\t")
        dataset_name = 'yelp'
    elif 'imdb' in dataset_name:
        df_db_val = pd.read_csv("../Datasets/sentiment_data/imdb/dev.tsv", sep="\t")
        dataset_name = 'imdb'
    elif 'enron' in dataset_name:
        df_db_val = pd.read_csv("../Datasets/spam_data/enron/dev.tsv", sep="\t").iloc[:3000]
        dataset_name = 'enron'
    else:
        raise ValueError(f"model_name must in twitter, jigsaw, amazon, yelp or enron")

    df_db_val = df_db_val.sample(500, random_state=2021)
    dataset = [i for i in zip(df_db_val.sentence, df_db_val.label)]
    dataset = textattack.datasets.Dataset(dataset)

    file_name = "{}_range{}_{}_{}adaptive{}.txt".format("da" if dynamic_attention else "",
                                           "".join([str(i) for i in da_range]) if dynamic_attention else "".join(
                                               [str(i) for i in fixed_range]),
                                           "dv" + str(decay_value),
                                           "dropout_" if dropout else "",
                                           str(adaptive_type))
    attack_args = AttackArgs(num_examples=500,
                             log_to_txt="adaptive_output/{}/{}/{}".format(model_name if model_name else model_dir,
                                                                          attack_method, file_name))
    attacker = Attacker(attack, dataset, attack_args)

    attack_results = attacker.attack_dataset()
