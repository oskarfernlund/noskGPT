"""
Training functions.
"""

from typing import Tuple

import torch

from noskgpt.model import LanguageModel


def get_batch(
    data: torch.tensor,
    block_size: int,
    batch_size: int,
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Fetch a batch of (context, target) data at random.
    Context (x) lags target (y) by 1 position in the sequence.

    :param data: Dataset from which to grab the batch.
    :param block_size: Maximum context length.
    :param batch_size: Batch size.
    :return: Tensors of context (x) and targets (y).
    """
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y


@torch.no_grad()
def estimate_loss(
    model: LanguageModel,
    train_data: torch.tensor,
    val_data: torch.tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
) -> dict:
    """
    Estimate training and validation loss by averaging over `eval_iters` batches.

    :param model: :class:`LanguageModel` to train.
    :param train_data: Training dataset.
    :param val_data: Validation dataset.
    :param block_size: Maximum context length.
    :param batch_size: Batch size.
    :param eval_iters: Number of batches to average over when reporting loss.
    :return: Dictionary containing training and validation loss values.
    """
    out = {}
    model.eval()
    for key, data in zip(["train", "val"], [train_data, val_data]):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size=block_size, batch_size=batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[key] = losses.mean()
    model.train()
    return out


def train_model(
    model: LanguageModel,
    train_data: torch.tensor,
    val_data: torch.tensor,
    learning_rate: float,
    block_size: int,
    batch_size: int,
    max_iters: int,
    eval_iters: int
) -> None:
    """
    Train the model using ADAM.

    :param model: :class:`LanguageModel` to train.
    :param train_data: Training dataset.
    :param val_data: Validation dataset.
    :param learning_rate: ADAM learning rate.
    :param block_size: Maximum context length.
    :param batch_size: Batch size.
    :param max_iters: Maximum number of optimisation steps.
    :param eval_iters: Number of batches to average over when reporting loss.
    """
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        if iter % eval_iters == 0:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                block_size=block_size,
                batch_size=batch_size,
                eval_iters=eval_iters,
            )
            print(
                f"step {iter}: ",
                f"train loss {losses['train']:.4f}, ",
                f"validation loss {losses['val']:.4f}"
            )

        x, y = get_batch(train_data, block_size=block_size, batch_size=batch_size)
        _, loss = model(x, y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()