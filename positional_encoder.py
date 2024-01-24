import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_size: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_size = seq_size
        self.positional_encodings = self._build_positional_encoding()

    def __call__(self, embeddings):
        # Scale the embeddings values
        x = embeddings / math.sqrt(self.d_model)

        # Pad and trim the sequence so that it has uniform length
        x = self._pad(self._trim(x))
        return x + self.positional_encodings

    def _build_positional_encoding(self):
        # divider term : 10000 ^ (i / d_model) - where i is the index in embedding vector
        divider = mx.exp((mx.arange(self.d_model) / self.d_model) * math.log(10000.0))

        # each position as a vector of dim d_model divided by the divider term
        positional_encodings = mx.array(np.empty((self.seq_size, self.d_model)))

        for pos in mx.arange(self.seq_size):
            positional_encodings[pos] = mx.full(self.d_model, pos) / divider

        # Apply sine on pair indices and cosine on impair ones
        positional_encodings[::2] = mx.sin(positional_encodings[::2])
        positional_encodings[1::2] = mx.cos(positional_encodings[1::2])

        return positional_encodings

    def _trim(self, embeddings):
        # Trim the sequence to a fixed length
        if len(embeddings) > self.seq_size:
            return embeddings[: self.seq_size, :]

        return embeddings

    def _pad(self, embeddings):
        # if the sequence is smaller than max size, we pad the space at the end of the sequence with zeros
        if len(embeddings) < self.seq_size:
            return mx.pad(
                embeddings,
                ((0, self.seq_size - len(embeddings)), (0, 0)),
            )

        return embeddings
