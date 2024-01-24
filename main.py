import spacy
from spacy.tokens import Token
from data_loader import ds
import mlx.core as mx
from transformer import Transformer


def to_tensor(token: Token):
    return mx.array(token.tensor)


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
