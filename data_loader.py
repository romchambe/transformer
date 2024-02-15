from datasets import load_dataset
import spacy
import mlx.core as mx


nlp = spacy.load("fr_core_news_sm")

ds = load_dataset(
    "parquet",
    data_files={"train": "data/wikipedia-0.parquet"},
)


def build_batches(data, batch_size):
    tensor = nlp(data).tensor
    nb_of_tokens = tensor.shape[0]
    return [
        mx.array(tensor[i : i + batch_size, :])
        for i in range(0, nb_of_tokens // batch_size)
    ]
