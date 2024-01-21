from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files={"train": "data/wikipedia-0.parquet"},
)
