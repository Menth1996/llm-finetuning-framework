import os
import argparse
from datasets import load_dataset
from transformers import TrainingArguments

from src.data_processor import LLMDataProcessor
from src.model_builder import LLMModelBuilder
from src.trainer import LLMTrainer
from src.utils import load_config

def main(config_path):
    config = load_config(config_path)

    # Data Processing
    data_processor = LLMDataProcessor(
        tokenizer_name=config["model"]["tokenizer_name"],
        max_length=config["data"]["max_length"]
    )
    dataset = load_dataset(config["data"]["dataset_name"])
    tokenized_datasets = data_processor.load_and_tokenize_data(
        dataset_name=config["data"]["dataset_name"],
        split=config["data"]["split"]
    )

    # Model Building
    model_builder = LLMModelBuilder(
        model_name=config["model"]["model_name"],
        num_labels=config["model"]["num_labels"]
    )
    model = model_builder.build_model()

    # Trainer
    llm_trainer = LLMTrainer(
        model=model,
        tokenizer=data_processor.tokenizer,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        output_dir=config["training"]["output_dir"],
        logging_dir=config["training"]["logging_dir"],
        report_to=config["training"]["report_to"]
    )
    llm_trainer.train(
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"]
    )

    print("LLM fine-tuning process completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM fine-tuning.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)
