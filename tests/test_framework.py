import unittest
import os
from unittest.mock import patch, MagicMock

from src.data_processor import LLMDataProcessor
from src.model_builder import LLMModelBuilder
from src.trainer import LLMTrainer
from src.utils import load_config, save_config

class TestLLMFinetuningFramework(unittest.TestCase):

    def setUp(self):
        self.test_config_path = "test_config.yaml"
        self.test_config_content = {
            "model": {
                "model_name": "bert-base-uncased",
                "tokenizer_name": "bert-base-uncased",
                "num_labels": 2
            },
            "data": {
                "dataset_name": "imdb",
                "split": "train",
                "max_length": 512
            },
            "training": {
                "output_dir": "./test_results",
                "logging_dir": "./test_logs",
                "report_to": "none",
                "num_train_epochs": 1,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "warmup_steps": 10,
                "weight_decay": 0.01
            }
        }
        save_config(self.test_config_content, self.test_config_path)

    def tearDown(self):
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        if os.path.exists("./test_results"):
            os.system("rm -rf ./test_results")
        if os.path.exists("./test_logs"):
            os.system("rm -rf ./test_logs")

    def test_load_config(self):
        config = load_config(self.test_config_path)
        self.assertEqual(config["model"]["model_name"], "bert-base-uncased")

    @patch("src.data_processor.AutoTokenizer.from_pretrained")
    @patch("src.data_processor.datasets.load_dataset")
    def test_data_processor(self, mock_load_dataset, mock_from_pretrained):
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.map.return_value = {"train": MagicMock()}
        mock_load_dataset.return_value = mock_dataset

        processor = LLMDataProcessor("bert-base-uncased")
        tokenized_data = processor.load_and_tokenize_data("imdb")
        self.assertIsNotNone(tokenized_data)

    @patch("src.model_builder.AutoModelForSequenceClassification.from_pretrained")
    def test_model_builder(self, mock_from_pretrained):
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        builder = LLMModelBuilder("bert-base-uncased", num_labels=2)
        model = builder.build_model()
        self.assertIsNotNone(model)

    @patch("src.trainer.Trainer")
    @patch("src.trainer.TrainingArguments")
    def test_llm_trainer(self, mock_training_arguments, mock_trainer):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_train_dataset = MagicMock()
        mock_eval_dataset = MagicMock()

        trainer_instance = LLMTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            output_dir="./test_results",
            logging_dir="./test_logs"
        )
        trainer_instance.train()
        mock_trainer.assert_called_once()

if __name__ == "__main__":
    unittest.main()
