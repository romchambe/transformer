import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        batch_dim: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.batch_dim = batch_dim
        self.positional_encodings = self._build_positional_encoding()

    def __call__(self, embeddings):
        # Scale the embeddings values
        x = embeddings / math.sqrt(self.embedding_dim)

        return x + self.positional_encodings

    def _build_positional_encoding(self):
        # divider term : 10000 ^ (i / embedding_dim) - where i is the index in embedding vector
        divider = mx.exp(
            (mx.arange(self.embedding_dim) / self.embedding_dim) * math.log(10000.0)
        )

        # each position as a vector of dim embedding_dim divided by the divider term
        positional_encodings = mx.array(np.empty((self.batch_dim, self.embedding_dim)))

        for pos in mx.arange(self.batch_dim):
            positional_encodings[pos] = mx.full(self.embedding_dim, pos) / divider

        # Apply sine on pair indices and cosine on impair ones
        positional_encodings[::2] = mx.sin(positional_encodings[::2])
        positional_encodings[1::2] = mx.cos(positional_encodings[1::2])

        return positional_encodings
