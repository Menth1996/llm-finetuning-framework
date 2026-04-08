# LLM Fine-tuning Framework

## A Scalable Python Framework for Fine-tuning Large Language Models

![LLM Fine-tuning](https://miro.medium.com/v2/resize:fit:1400/1*tX_201_6-6-Q-X-P-Q-X-Q.png)

This repository presents a comprehensive and scalable Python framework designed for efficient fine-tuning of Large Language Models (LLMs). It provides a structured and modular approach to adapt pre-trained LLMs to specific downstream tasks and domain-specific datasets, ensuring high performance, reproducibility, and ease of experimentation. The framework integrates with popular libraries like HuggingFace Transformers and PyTorch, and supports distributed training strategies.

## Features

- **Modular Design**: Easily extendable components for data loading, model definition, training loops, and evaluation.
- **LLM Compatibility**: Supports a wide range of pre-trained LLMs from the HuggingFace ecosystem.
- **Distributed Training**: Seamless integration with PyTorch Distributed (DDP, FSDP) and Accelerate for multi-GPU and multi-node training.
- **Configuration Management**: YAML-based configuration for managing hyperparameters, model architectures, and training settings.
- **Experiment Tracking**: Built-in support for MLflow and Weights & Biases for logging metrics, parameters, and model checkpoints.
- **Data Pipelines**: Efficient data loading, preprocessing, and tokenization pipelines for various text datasets.
- **Evaluation Suite**: Comprehensive evaluation metrics and tools for assessing fine-tuned model performance.

## Installation

To get started with the LLM Fine-tuning Framework, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Menth1996/llm-finetuning-framework.git
cd llm-finetuning-framework
pip install -r requirements.txt
```

## Usage

### Fine-tuning an LLM on a Custom Dataset

```python
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

# 1. Load configuration
# config = load_config("configs/default_config.yaml") # Placeholder for actual config loading

# 2. Load dataset (example with IMDB for sentiment analysis)
dataset = load_dataset("imdb")

# 3. Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 5. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="mlflow" # Integrate with MLflow
)

# 6. Initialize and train the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

print("LLM fine-tuning complete!")
```

## Project Structure

```
llm-finetuning-framework/
├── src/
│   ├── __init__.py
│   ├── data_processor.py     # Data loading, cleaning, and tokenization
│   ├── model_builder.py      # Model loading and architecture definition
│   ├── trainer.py            # Core training and evaluation logic
│   └── utils.py              # Utility functions (e.g., config management)
├── configs/
│   ├── default_config.yaml   # Default training configuration
│   └── fsdp_config.yaml      # FSDP-specific configurations
├── data/
│   └── .gitkeep              # Placeholder for datasets
├── scripts/
│   └── train.py              # Main script to run fine-tuning
├── tests/
│   └── test_framework.py     # Unit tests for framework components
├── requirements.txt          # Python dependencies
├── Dockerfile                # Dockerfile for containerization
├── README.md                 # Project overview and documentation
└── LICENSE                   # License file
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines on how to submit pull requests, report bugs, and suggest new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
