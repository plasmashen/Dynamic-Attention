
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)

from model.sequence_classification import (
    BertPrefixForSequenceClassification,
    BertPromptForSequenceClassification
)

#
# PREFIX_MODELS = {
#     "bert": {
#         TaskType.SEQUENCE_CLASSIFICATION: BertPrefixForSequenceClassification
#     },
# }
#
# PROMPT_MODELS = {
#     "bert": {
#         TaskType.SEQUENCE_CLASSIFICATION: BertPromptForSequenceClassification
#     },
# }
#
# AUTO_MODELS = {
#     TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
#     TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
# }


def get_model(model_args, config: AutoConfig, fix_bert: bool = False):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size

        model_class = BertPrefixForSequenceClassification
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    elif model_args.prompt:
        config.pre_seq_len = model_args.pre_seq_len
        model_class = BertPromptForSequenceClassification
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    else:
        model_class = AutoModelForSequenceClassification
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )

        bert_param = 0
        if fix_bert:
            for param in model.bert.parameters():
                param.requires_grad = False
            for _, param in model.bert.named_parameters():
                bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model
