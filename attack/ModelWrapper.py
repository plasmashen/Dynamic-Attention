"""
HuggingFace Model Wrapper
--------------------------
"""

import torch
import transformers
from torch.nn import CrossEntropyLoss
import numpy as np
import textattack

from textattack.models.wrappers import ModelWrapper
from textattack.models.tokenizers.t5_tokenizer import T5Tokenizer

torch.cuda.empty_cache()


class PyTorchModelWrapper(ModelWrapper):
    """Loads a PyTorch model (`nn.Module`) and tokenizer.

    Args:
        model (torch.nn.Module): PyTorch model
        tokenizer: tokenizer whose output can be packed as a tensor and passed to the model.
            No type requirement, but most have `tokenizer` method that accepts list of strings.
    """

    def __init__(self, model, tokenizer, decay_value, dynamic_attention, da_range=[3, 8], dropout=False):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.decay_value = decay_value
        self.dynamic_attention = dynamic_attention
        self.da_range = da_range
        self.dropout = dropout

    def to(self, device):
        self.model.to(device)

    def __call__(self, text_input_list, batch_size=4):
        model_device = next(self.model.parameters()).device
        ids = self.tokenizer(text_input_list, padding=True)
        if isinstance(self.tokenizer, T5Tokenizer):
            ids = torch.tensor(ids["input_ids"]).to(model_device)
        else:
            ids = torch.tensor(ids).to(model_device)
        if self.dropout:
            self.model.train()
        with torch.no_grad():
            outputs = batch_model_predict(
                self.model, ids, decay_value=self.decay_value,
                random_top=self.dynamic_attention, random_bound=self.da_range, batch_size=batch_size
            )

        return outputs

    def get_grad(self, text_input, loss_fn=CrossEntropyLoss()):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
            loss_fn (torch.nn.Module): loss function. Default is `torch.nn.CrossEntropyLoss`
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )
        if not isinstance(loss_fn, torch.nn.Module):
            raise ValueError("Loss function must be of type `torch.nn.Module`.")

        self.model.train()

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        ids = self.tokenizer([text_input])
        ids = torch.tensor(ids).to(model_device)

        predictions = self.model(ids)

        output = predictions.argmax(dim=1)
        loss = loss_fn(predictions, output)
        loss.backward()

        # grad w.r.t to word embeddings
        grad = torch.transpose(emb_grads[0], 0, 1)[0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": ids[0].tolist(), "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [self.tokenizer.convert_ids_to_tokens(self.tokenizer(x)) for x in inputs]


class HuggingFaceModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, long_text, decay_value, dynamic_attention, da_range=[3, 6], dropout=False):
        assert isinstance(
            model, transformers.PreTrainedModel
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.long_text = long_text
        self.decay_value = decay_value
        self.dynamic_attention = dynamic_attention
        self.da_range = da_range
        self.dropout = dropout

    def __call__(self, text_input_list, batch_size=4):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            256
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)
        if self.dropout:
            self.model.train()
        if self.da_range[1] < 1:
            text_length = sum([len(text_input.split(' ')) for text_input in text_input_list]) / len(text_input_list)
            if self.long_text:
                top1 = min(int(self.da_range[0] * text_length), 5)
                top2 = min(max(int(self.da_range[1] * text_length), 5), 20)
            else:
                top1 = int(self.da_range[0] * text_length)
                top2 = int(self.da_range[1] * text_length)
        else:
            top1 = int(self.da_range[0])
            top2 = int(self.da_range[1])

        with torch.no_grad():
            outputs = batch_model_predict_2(
                self.model, **inputs_dict, decay_value=self.decay_value,
                random_top=self.dynamic_attention, random_bound=[top1, top2], batch_size=batch_size
            )

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]


class HuggingFaceModelWrapperForGPT(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, long_text, decay_value, dynamic_attention, da_range=[3, 6], dropout=False):
        assert isinstance(
            model, transformers.PreTrainedModel
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.long_text = long_text
        self.decay_value = decay_value
        self.dynamic_attention = dynamic_attention
        self.da_range = da_range
        self.dropout = dropout

    def __call__(self, text_input_list, batch_size=4):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            256
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)
        # if self.dropout:
        #     self.model.train()
        if self.dropout:
            self.model.train()
        if self.da_range[1] < 1:
            text_length = sum([len(text_input.split(' ')) for text_input in text_input_list]) / len(text_input_list)
            if self.long_text:
                top1 = min(int(self.da_range[0] * text_length), 5)
                top2 = min(max(int(self.da_range[1] * text_length), 5), 20)
            else:
                top1 = int(self.da_range[0] * text_length)
                top2 = int(self.da_range[1] * text_length)
        else:
            top1 = int(self.da_range[0])
            top2 = int(self.da_range[1])

        with torch.no_grad():
            outputs = batch_model_predict_1(
                self.model, **inputs_dict, decay_value=self.decay_value,
                random_top=self.dynamic_attention, random_bound=[top1, top2], batch_size=batch_size
            )

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]


def batch_model_predict(model_predict, inputs, decay_value,
                        random_top, random_bound, batch_size=32):
    """Runs prediction on iterable ``inputs`` using batch size ``batch_size``.

    Aggregates all predictions into an ``np.ndarray``.
    """
    outputs = []
    i = 0
    while i < len(inputs):
        batch = inputs[i: i + batch_size]
        batch_preds = model_predict(batch, decay_value=decay_value, random_top=random_top,
                                    random_bound=random_bound)

        # Some seq-to-seq models will return a single string as a prediction
        # for a single-string list. Wrap these in a list.
        if isinstance(batch_preds, str):
            batch_preds = [batch_preds]

        # Get PyTorch tensors off of other devices.
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu()

        # Cast all predictions iterables to ``np.ndarray`` types.
        if not isinstance(batch_preds, np.ndarray):
            batch_preds = np.array(batch_preds)
        outputs.append(batch_preds)
        i += batch_size

    return np.concatenate(outputs, axis=0)


def batch_model_predict_1(model_predict, input_ids, attention_mask, decay_value,
                          random_top, random_bound, batch_size=32):
    """Runs prediction on iterable ``inputs`` using batch size ``batch_size``.

    Aggregates all predictions into an ``np.ndarray``.
    """
    outputs = []
    i = 0
    while i < len(input_ids):
        batch_1 = input_ids[i: i + batch_size]
        batch_2 = attention_mask[i: i + batch_size]
        batch_preds = model_predict(input_ids=batch_1, attention_mask=batch_2, decay_value=decay_value,
                                    random_top=random_top, random_bound=random_bound).logits

        # Some seq-to-seq models will return a single string as a prediction
        # for a single-string list. Wrap these in a list.
        if isinstance(batch_preds, str):
            batch_preds = [batch_preds]

        # Get PyTorch tensors off of other devices.
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu()

        # Cast all predictions iterables to ``np.ndarray`` types.
        if not isinstance(batch_preds, np.ndarray):
            batch_preds = np.array(batch_preds)
        outputs.append(batch_preds)
        i += batch_size

    return np.concatenate(outputs, axis=0)


def batch_model_predict_2(model_predict, input_ids, token_type_ids, attention_mask, decay_value,
                          random_top, random_bound, batch_size=32):
    """Runs prediction on iterable ``inputs`` using batch size ``batch_size``.

    Aggregates all predictions into an ``np.ndarray``.
    """
    outputs = []
    i = 0
    while i < len(input_ids):
        batch_1 = input_ids[i: i + batch_size]
        batch_2 = token_type_ids[i: i + batch_size]
        batch_3 = attention_mask[i: i + batch_size]
        batch_preds = model_predict(input_ids=batch_1, token_type_ids=batch_2, attention_mask=batch_3,
                                    decay_value=decay_value, random_top=random_top, random_bound=random_bound).logits

        # Some seq-to-seq models will return a single string as a prediction
        # for a single-string list. Wrap these in a list.
        if isinstance(batch_preds, str):
            batch_preds = [batch_preds]

        # Get PyTorch tensors off of other devices.
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu()

        # Cast all predictions iterables to ``np.ndarray`` types.
        if not isinstance(batch_preds, np.ndarray):
            batch_preds = np.array(batch_preds)
        outputs.append(batch_preds)
        i += batch_size

    return np.concatenate(outputs, axis=0)
