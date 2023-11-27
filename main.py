import spacy
from data_loader import ds


# parser = ArgumentParser()

# parser.add_argument(
#     "--dataset",
#     help="Select which dataset to use to train the model",
#     default="wikitext2",
#     choices=["ptb", "wikitext2", "wikitext103"],
# )

# args = parser.parse_args()


def main():
    # Load vocab and dataset:
    nlp = spacy.load("fr_core_news_sm")

    doc = ds["train"][0]["text"]
    doc = nlp(doc)


main()
