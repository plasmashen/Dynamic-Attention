import argparse
import sys, re

sys.path.append("..")

from textattack import Attacker
from textattack import AttackArgs
from textattack import Attack
from textattack.constraints.overlap import LevenshteinEditDistance
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
from textattack.goal_functions import NonOverlappingOutput
from textattack.search_methods import GreedyWordSwapWIR
from textattack.goal_functions import MinimizeBleu
from textattack.search_methods import GreedySearch
from textattack.transformations import WordSwapInflections
from textattack.attack_recipes import AttackRecipe
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
from modeling_t5 import T5ForTextToText
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
import pandas as pd
# from torch.nn import CrossEntropyLoss
import textattack
from ModelWrapper import HuggingFaceModelWrapper, PyTorchModelWrapper


class Seq2SickCheng2018BlackBox(AttackRecipe):
    """Cheng, Minhao, et al.

    Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with
    Adversarial Examples

    https://arxiv.org/abs/1803.01128

    This is a greedy re-implementation of the seq2sick attack method. It does
    not use gradient descent.
    """

    @staticmethod
    def build(model_wrapper, max_rate=0.2, goal_function="non_overlapping"):
        #
        # Goal is non-overlapping output.
        #
        goal_function = NonOverlappingOutput(model_wrapper)
        transformation = WordSwapEmbedding(max_candidates=50)
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # In these experiments, we hold the maximum difference
        # on edit distance (ϵ) to a constant 30 for each sample.
        #
        constraints.append(LevenshteinEditDistance(30))
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(wir_method="unk")

        return Attack(goal_function, constraints, transformation, search_method)


class MorpheusTan2020(AttackRecipe):
    """Samson Tan, Shafiq Joty, Min-Yen Kan, Richard Socher.

    It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations

    https://www.aclweb.org/anthology/2020.acl-main.263/
    """

    @staticmethod
    def build(model_wrapper, max_rate=0.2):
        #
        # Goal is to minimize BLEU score between the model output given for the
        # perturbed input sequence and the reference translation
        #
        goal_function = MinimizeBleu(model_wrapper)

        # Swap words with their inflections
        transformation = WordSwapInflections()

        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        #
        # Greedily swap words (see pseudocode, Algorithm 1 of the paper).
        #
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)


class TextBuggerLi2018(AttackRecipe):

    @staticmethod
    def build(model_wrapper, max_rate=0.2):
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

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        goal_function = MinimizeBleu(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)


class TextFoolerJin2019(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate=0.2):
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
        goal_function = MinimizeBleu(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)


def AutoAttack(attack_method):
    if attack_method == "textbugger":
        return TextBuggerLi2018
    if attack_method == "textfooler":
        return TextFoolerJin2019
    if attack_method == "morpheus":
        return MorpheusTan2020
    if attack_method == "seq2sick":
        return Seq2SickCheng2018BlackBox


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_method", type=str, required=True, choices=["textbugger", "textfooler", "pwws", "bae",
                                                                             "deepwordbug", "pruthi", "checklist",
                                                                             "bert_attack", "a2t", "morpheus",
                                                                             "seq2sick"])
    parser.add_argument("--max_rate", default=0.2, type=float)
    parser.add_argument("--model_name", default="t5-en-fr", type=str)
    parser.add_argument("--decay_value", default=1, type=int)
    parser.add_argument("--dynamic_attention", action="store_true")
    parser.add_argument("--fixed_range", nargs='+', type=int, default=[])
    parser.add_argument("--da_range", nargs='+', type=int, default=[3, 8])
    parser.add_argument("--dropout", action="store_true")

    args = parser.parse_args()
    attack_method = args.attack_method
    max_rate = args.max_rate
    model_name = args.model_name
    if "sum" not in args.model_name:
        _, source_language, target_language = model_name.split("-")
    decay_value = args.decay_value
    dynamic_attention = args.dynamic_attention
    fixed_range = [int(i) for i in args.fixed_range]
    da_range = [int(i) for i in args.da_range]
    dropout = args.dropout
    model = T5ForTextToText.from_pretrained(model_name)
    model = PyTorchModelWrapper(model, model.tokenizer, decay_value, dynamic_attention, da_range, dropout)
    if "sum" in args.model_name:
        dataset = ("gigaword", None, "test")
    else:
        dataset = (
            "textattack.datasets.helpers.TedMultiTranslationDataset",
            source_language,
            target_language,
        )
    if "sum" in args.model_name:
        dataset = textattack.datasets.HuggingFaceDataset(
            *dataset, shuffle=False
        )
    else:
        dataset = eval(f"{dataset[0]}")(*dataset[1:])

    if "sum" in args.model_name:
        save_dir = 'adv_sum'
    else:
        save_dir = 'adv_nmt'

    attack_module = AutoAttack(attack_method)
    attack = attack_module.build(model, max_rate)

    file_name = "{}.txt".format(attack_method)
    attack_args = AttackArgs(num_examples=1000,
                             log_to_txt="{}/{}/{}".format(save_dir, model_name, file_name))
    attacker = Attacker(attack, dataset, attack_args)

    attack_results = attacker.attack_dataset()
