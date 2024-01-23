import spacy
from spacy.tokens import Token
from data_loader import ds
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from numpy.typing import NDArray
import math


def to_tensor(token: Token):
    return mx.array(token.tensor)


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
        x = self._pad(self._trim(embeddings))
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
        if len(embeddings) < self.seq_size:
            return mx.pad(
                embeddings,
                ((0, self.seq_size - len(embeddings)), (0, 0)),
            )

        return embeddings


class Encoder(nn.Module):
    def __init__(self, d_model: int, seq_size: int, dropout: float, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.positionally_encoded = PositionalEncoder(d_model, seq_size)

    def __call__(self, embeddings):
        pos_encoded_embeddings = self.positionally_encoded(embeddings)
        return pos_encoded_embeddings


class Transformer(nn.Module):
    def __init__(self, d_model: int, seq_size: int, dropout=0.2, num_heads: int = 3):
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(d_model, seq_size, dropout, num_heads)

    def __call__(self, sentence: NDArray[np.float32]):
        scaled_embeddings = sentence / math.sqrt(self.d_model)

        encoded = self.encoder(scaled_embeddings)
        return encoded


def main():
    # Load vocab and dataset:
    nlp = spacy.load("fr_core_news_sm")
    doc = ds["train"][0]["text"]
    doc = nlp(doc)
    d_model = len(doc.vector)

    # Parse sentences and turn them into tensors
    vectorized_sentences = [to_tensor(s) for s in doc.sents if s.__len__() >= 3]

    # Launch model
    model = Transformer(d_model, 64)
    output = model(vectorized_sentences[0])
    print(output.shape, output[41], output[42])

    # for epoch in range(epochs):
    # for data, target in data_loader:
    #     optimizer.zero_grad()
    #     output = model(data)
    #     loss = loss_fn(output, target)
    #     loss.backward()
    #     optimizer.step()

    # for sentence in sents:
    #     if sentence.__len__() > 3:
    #         print(len(sentence))


main()
