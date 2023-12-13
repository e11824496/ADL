from typing import Union

import torch
from torch import Tensor

from model import Embedding


class Model(torch.nn.Module):
    def __init__(self,
                 num_classes: int,
                 mean_amount: float,
                 std_amount: float) -> None:
        """
        Initializes the model.

        Args:
            num_classes (int): The number of classes for classification.
            mean_amount (float): The mean amount.
            std_amount (float): The standard deviation amount.
        """
        super().__init__()
        self.mean_amount = mean_amount
        self.std_amount = std_amount

        self.embedding = Embedding()
        self.embedding_dim = 768
        self.num_classes = num_classes
        self.fc1 = torch.nn.Linear(self.embedding_dim + 1, 256)
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

        x = [f'Time: {time}, Description: {description}'
             for time, amount, description in zip(time, amount, description)]
        x = self.embedding(x)

        amount = (amount - self.mean_amount) / self.std_amount
        amount = amount.to(self.embedding.model.device)
        x = torch.cat((x, amount.unsqueeze(1)), dim=1)

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
