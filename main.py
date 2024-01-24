import spacy
from spacy.tokens import Token
from data_loader import ds
import mlx.core as mx
import mlx.nn as nn
import numpy as np
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, seq_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.seq_size = seq_size
        self.d_model = d_model

        # Each linear layer will hold the "heads" concatenated
        input_dim = num_heads * seq_size
        output_dim = num_heads * seq_size // 2

        self.query_proj = nn.Linear(input_dim, output_dim)
        self.key_proj = nn.Linear(input_dim, output_dim)
        self.val_proj = nn.Linear(input_dim, output_dim)

        self.out_proj = nn.Linear(output_dim, seq_size)

    def __call__(self, x):
        multi_head_input = mx.concatenate([x for _ in range(self.num_heads)]).T

        linear_output_shape = [
            self.num_heads,
            self.seq_size // 2,
            self.d_model,
        ]

        query = self.query_proj(multi_head_input).reshape(linear_output_shape)
        key = self.key_proj(multi_head_input).reshape(linear_output_shape)
        value = self.val_proj(multi_head_input).reshape(linear_output_shape)

        attention_filter = mx.softmax(
            mx.matmul(query, key.transpose(0, 2, 1)) / math.sqrt(self.seq_size), axis=-1
        )

        filtered_value = mx.matmul(attention_filter, value).reshape(
            self.seq_size * self.num_heads // 2, self.d_model
        )

        return self.out_proj(filtered_value.T).T


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_size: int,
        dropout: float,
        num_heads: int,
    ):
        super().__init__()
        self.attention_heads = MultiHeadAttention(d_model, seq_size, num_heads)
        self.attention_add_norm = [nn.Dropout(dropout), nn.LayerNorm(d_model)]

    def __call__(self, x):
        output = self.attention_heads(x)

        for layer in self.attention_add_norm:
            output = layer(output)

        output = output + x

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
