"""
Classes defining our language model architecture.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from noskgpt.tokeniser import CharToIntTokeniser


class AttentionHead(nn.Module):
    """
    Scaled dot-product attention head.
    """

    def __init__(
        self,
        block_size: int,
        num_embeddings: int,
        head_size: int,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialise attention head components.

        :param block_size: Maximum context length.
        :param num_embeddings: Embedding dimensionality.
        :param head_size: Size of the attention head.
        :param dropout: Proportion of neurons to mask during training, defaults to 0.0.
        """
        super().__init__()
        self.key = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Attention head forward pass.

        :param x: Input tensor (B, T, C).
        :return: Output tensor (B, T, `head_size`).
        """
        _, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # Compute attention score / affinities
        w = q @ k.transpose(-2, -1) * C ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        # Perform weighted aggregation of values
        v = self.value(x)
        return w @ v
    

class MultiHeadAttention(nn.Module):
    """
    Multiple scaled dot-product attention heads in parallel.
    """

    def __init__(
        self,
        block_size: int,
        num_embeddings: int,
        num_heads: int,
        head_size: int,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialise multiple scaled dot product attention heads in parallel.

        :param block_size: Maximum context length.
        :param num_embeddings: Embedding dimensionality.
        :param num_heads: Number of attention heads.
        :param head_size: Size of the attention heads.
        :param dropout: Proportion of neurons to mask during training, defaults to 0.0.
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    block_size=block_size,
                    num_embeddings=num_embeddings,
                    head_size=head_size,
                    dropout=dropout,
                ) for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(num_embeddings, num_embeddings)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Multi-head attention head forward pass.

        :param x: Input tensor (B, T, C).
        :return: Output tensor (B, T, C).
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.projection(out)


class FeedForward(nn.Module):
    """
    Simple feedforward neural network.
    """

    def __init__(self, num_embeddings: int, dropout: float = 0.0) -> None:
        """
        Initialise feedforward neural network components.

        :param num_embeddings: Embedding dimensionality.
        :param dropout: Proportion of neurons to mask during training, defaults to 0.0.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings),
            nn.ReLU(),
            nn.Linear(4 * num_embeddings, num_embeddings),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Feedforward neural network forward pass.

        :param x: Input tensor (B, T, C).
        :return: Output tensor (B, T, C).
        """
        return self.network(x)
    

class TransformerBlock(nn.Module):
    """
    Transformer block.
    """

    def __init__(
        self,
        block_size: int,
        num_embeddings: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialise components of the transformer block.

        :param block_size: Maximum context length.
        :param num_embeddings: Embedding dimensionality.
        :param num_heads: Number of attention heads.
        :param dropout: Proportion of neurons to mask during training, defaults to 0.0.
        """
        super().__init__()
        assert num_embeddings % num_heads == 0, (
            "`num_embeddings` must be a multiple of `num_heads`."
        )
        head_size = num_embeddings // num_heads
        self.self_attention = MultiHeadAttention(
            block_size=block_size,
            num_embeddings=num_embeddings,
            num_heads=num_heads,
            head_size=head_size,
            dropout=dropout,
        )
        self.feedforward = FeedForward(
            num_embeddings=num_embeddings, dropout=dropout
        )
        self.layernorm1 = nn.LayerNorm(num_embeddings)
        self.layernorm2 = nn.LayerNorm(num_embeddings)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Transformer block forward pass.

        :param x: Input tensor (B, T, C).
        :return: Output tensor (B, T, C).
        """
        x = x + self.self_attention(self.layernorm1(x))
        return x + self.feedforward(self.layernorm2(x))
    

class LanguageModel(nn.Module):
    """
    Transformer-based generative language model.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        num_embeddings: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """
        Initialise language model components.

        :param vocab_size: Size of the model vocabulary.
        :param block_size: Maximum context length.
        :param num_embeddings: Embedding dimensionality.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of transformer blocks.
        :param dropout: Proportion of neurons to mask during training, defaults to 0.0.
        """
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, num_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, num_embeddings)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    block_size=block_size,
                    num_embeddings=num_embeddings,
                    num_heads=num_heads,
                    dropout=dropout
                ) for _ in range(num_layers)
            ]
        )
        self.feedforward = FeedForward(num_embeddings=num_embeddings, dropout=dropout)
        self.output_head = nn.Linear(num_embeddings, vocab_size)

    def forward(
        self, idx: torch.tensor, targets: Optional[torch.tensor] = None
    ) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        """
        Language model forward pass.

        :param idx: Embedding lookup index.
        :param targets: Optional target logits, defaults to None.
        :return: Predicted logits, or logits and loss (if targets are provided).
        """
        assert idx.ndim == 2, "`idx` must have 2 dimensions."
        B, T = idx.shape
        
        # Compute logits
        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(torch.arange(T))
        x = token_embedding + position_embedding
        x = self.transformer_blocks(x)
        x = self.feedforward(x)
        logits = self.output_head(x)

        # Compute cross-entropy loss if targets were provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) # ()

        return logits, loss
    
    def generate(
        self,
        prompt: str,
        tokeniser: CharToIntTokeniser,
        max_new_tokens: int
    ) -> None:
        """
        Generate new text!

        :param prompt: Text prompt.
        :param tokeniser: Tokeniser to decode the model output.
        :param max_new_tokens: Maximum number of new tokens to generate.
        """
        idx = tokeniser.encode(prompt).view(1, -1)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            print(tokeniser.decode(idx_next), end="")
        print("\n")