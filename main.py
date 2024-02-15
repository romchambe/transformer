from spacy.tokens import Token
from data_loader import ds, nlp, build_batches
import mlx.core as mx
from transformer import Transformer
from functools import reduce


def main():
    # Load the first ten articles
    corpus = reduce(lambda store, text: store + text, ds["train"][:10]["text"])

    # initialize main params
    vocab_dim = len(nlp.vocab)
    embedding_dim = 96
    batch_dim = 128

    batches = build_batches(corpus, batch_dim)
    model = Transformer(embedding_dim, batch_dim, vocab_dim)

    # Launch model
    for b in batches:
        output = model(b)
        print(output.shape, output)

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
