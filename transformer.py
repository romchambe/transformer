import mlx.core as mx
import mlx.nn as nn
import math
from positional_encoder import PositionalEncoder


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, seq_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.seq_size = seq_size
        self.d_model = d_model

        # Each linear layer will hold the "heads" concatenated (hence input of dim: num_heads * seq_size)
        # We arbitrarily chose to divide the output size by a factor of 2
        input_dim = num_heads * seq_size
        output_dim = num_heads * seq_size // 2

        self.query_proj = nn.Linear(input_dim, output_dim)
        self.key_proj = nn.Linear(input_dim, output_dim)
        self.val_proj = nn.Linear(input_dim, output_dim)

        # Linear layer that will condense the concatenated output of all heads
        self.out_proj = nn.Linear(output_dim, seq_size)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        # The input is copied to all heads
        multi_head_input = mx.concatenate([x for _ in range(self.num_heads)]).T

        linear_proj_output_shape = [
            self.num_heads,
            self.seq_size // 2,
            self.d_model,
        ]

        # We project three matrices
        query = self.query_proj(multi_head_input).reshape(linear_proj_output_shape)
        key = self.key_proj(multi_head_input).reshape(linear_proj_output_shape)
        value = self.val_proj(multi_head_input).reshape(linear_proj_output_shape)

        # We build an attention filter based on cosine similarity between query and key, softmax activated
        attention_filter = mx.softmax(
            mx.matmul(query, key.transpose(0, 2, 1)) / math.sqrt(self.seq_size), axis=-1
        )

        # Attention filter is applied to the value
        filtered_value = mx.matmul(attention_filter, value).reshape(
            self.seq_size * self.num_heads // 2, self.d_model
        )

        return self.dropout(self.out_proj(filtered_value.T).T)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_size: int,
        dropout: float,
        num_heads: int,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, seq_size, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward = [
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        ]
        self.ff_norm = nn.LayerNorm(d_model)

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
        d_model: int,
        seq_size: int,
        dropout: float,
        encoder_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.positional_encoder = PositionalEncoder(d_model, seq_size)
        self.encoder_layers = [
            EncoderLayer(d_model, seq_size, dropout, num_heads)
            for _ in range(encoder_layers)
        ]

    def __call__(self, embeddings):
        x = self.positional_encoder(embeddings)
        for layer in self.encoder_layers:
            x = layer(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_size: int,
        dropout: float = 0.2,
        encoder_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        if (d_model % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({d_model} % {num_heads}) != 0"
            )

        self.d_model = d_model
        self.encoder = Encoder(d_model, seq_size, dropout, encoder_layers, num_heads)

    def __call__(self, seq):
        encoded = self.encoder(seq)
        return encoded
