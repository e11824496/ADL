from typing import Any, Union
import torch.nn.functional as F
from torch import nn

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class Embedding(nn.Module):
    def __init__(self) -> None:
        """
        Initializes the Embedding module.

        This module uses a pre-trained tokenizer and model from the
        'intfloat/multilingual-e5-base' model.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            'intfloat/multilingual-e5-base')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

    def forward(self, input_texts) -> Any:
        """
        Performs forward pass of the Embedding module.

        Args:
            input_texts (str or List[str]): Input text(s) to be embedded.

        Returns:
            torch.Tensor: Embeddings of the input text(s).
        """
        if not isinstance(input_texts, list):
            input_texts = [input_texts]

        # Tokenize the input texts
        batch_dict = self.tokenizer(
            input_texts, max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt')

        batch_dict = {k: v.to(self.model.device)
                      for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class Model(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        """
        Initializes the model.

        Args:
            num_classes (int): The number of classes for classification.
        """
        super().__init__()

        self.embedding = Embedding()
        self.embedding_dim = 768
        self.num_classes = num_classes
        self.fc1 = torch.nn.Linear(self.embedding_dim, 256)
        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, self.num_classes)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self,
                description: Union[str, list[str]],
                time: Union[str, list[str]],
                amount: Union[str, list[str]],
                label: Union[int, list[int]] = None) -> Tensor:
        """
        Forward pass of the model.

        Args:
            description (Union[str, list[str]]): The description input.
            time (Union[str, list[str]]): The time input.
            amount (Union[str, list[str]]): The amount input.
            label (Union[int, list[int]], optional): The label input.
                Defaults to None.

        Returns:
            dict: A dictionary containing the logits and loss
                (if label is not None).
        """
        if isinstance(description, str):
            description = [description]
        if isinstance(time, str):
            time = [time]
        if isinstance(amount, float):
            amount = [amount]
        if isinstance(time, str):
            time = [time]

        x = [f'Time: {time}, Amount: {amount}, Description: {description}'
             for time, amount, description in zip(time, amount, description)]
        x = self.embedding(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        loss = None
        if label is not None:
            label = label.to(self.embedding.model.device)
            loss = self.loss_fn(x, label)

        return {
            'logits': x,
            'loss': loss
        }
