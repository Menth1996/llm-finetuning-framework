import datasets
from transformers import AutoTokenizer

class LLMDataProcessor:
    def __init__(self, tokenizer_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def load_and_tokenize_data(self, dataset_name: str, split: str = "train"):
        dataset = datasets.load_dataset(dataset_name, split=split)
        tokenized_dataset = dataset.map(self._tokenize_function, batched=True, remove_columns=dataset.column_names)
        return tokenized_dataset

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)
