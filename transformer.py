from typing import Any
import mlx.core as mx
import mlx.nn as nn
import math
from positional_encoder import PositionalEncoder


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embedding_dim: int, batch_dim: int, num_heads: int, dropout: float
    ):
        super().__init__()
        self.num_heads = num_heads
        self.batch_dim = batch_dim
        self.embedding_dim = embedding_dim

        # Each linear layer will hold the "heads" concatenated (hence input of dim: num_heads * batch_dim)
        # We arbitrarily chose to divide the output size by a factor of 2
        input_dim = num_heads * batch_dim
        output_dim = num_heads * batch_dim // 2

        self.query_proj = nn.Linear(input_dim, output_dim)
        self.key_proj = nn.Linear(input_dim, output_dim)
        self.val_proj = nn.Linear(input_dim, output_dim)

        # Linear layer that will condense the concatenated output of all heads
        self.out_proj = nn.Linear(output_dim, batch_dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        # The input is copied to all heads

        multi_head_input = mx.concatenate([x for _ in range(self.num_heads)]).T

        linear_proj_output_shape = [
            self.num_heads,
            self.batch_dim // 2,
            self.embedding_dim,
        ]

        # We project three matrices
        query = self.query_proj(multi_head_input).reshape(linear_proj_output_shape)
        key = self.key_proj(multi_head_input).reshape(linear_proj_output_shape)
        value = self.val_proj(multi_head_input).reshape(linear_proj_output_shape)

        # We build an attention filter based on cosine similarity between query and key, softmax activated
        attention_filter = mx.softmax(
            mx.matmul(query, key.transpose(0, 2, 1)) / math.sqrt(self.batch_dim),
            axis=-1,
        )

        # Attention filter is applied to the value
        filtered_value = mx.matmul(attention_filter, value).reshape(
            self.batch_dim * self.num_heads // 2, self.embedding_dim
        )

        return self.dropout(self.out_proj(filtered_value.T).T)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        batch_dim: int,
        dropout: float,
        num_heads: int,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            embedding_dim, batch_dim, num_heads, dropout
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = [
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        ]
        self.ff_norm = nn.LayerNorm(embedding_dim)

    def __call__(self, x):
        output = self.attention(x)

        # original input is reinjected - residual connection
        output = self.attention_norm(output + x)

        for layer in self.feed_forward:
            output = layer(output)

        output = self.ff_norm(output + x)

        return output


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        batch_dim: int,
        dropout: float,
        encoder_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.positional_encoder = PositionalEncoder(embedding_dim, batch_dim)
        self.encoder_layers = [
            EncoderLayer(embedding_dim, batch_dim, dropout, num_heads)
            for _ in range(encoder_layers)
        ]

    def __call__(self, embeddings):
        x = self.positional_encoder(embeddings)

        for layer in self.encoder_layers:
            x = layer(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        batch_dim: int,
        vocab_dim: int,
        dropout: float,
        num_heads: int,
    ):
        super().__init__()
        self.positional_encoder = PositionalEncoder(embedding_dim, batch_dim)

    def __call__(self, encoded, target):
        # Masquer la target

        print(target)


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        batch_dim: int,
        vocab_dim: int,
        dropout: float = 0.2,
        encoder_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()

        if (embedding_dim % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({embedding_dim} % {num_heads}) != 0"
            )

        self.embedding_dim = embedding_dim
        self.encoder = Encoder(
            embedding_dim, batch_dim, dropout, encoder_layers, num_heads
        )
        self.decoder = Decoder(embedding_dim, batch_dim, vocab_dim, dropout, num_heads)

    def __call__(self, seq):
        encoded = self.encoder(seq)

        return encoded
