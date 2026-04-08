from transformers import Trainer, TrainingArguments
import torch

class LLMTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, output_dir, logging_dir, report_to="none"):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.report_to = report_to

    def train(self, num_train_epochs=3, per_device_train_batch_size=8, per_device_eval_batch_size=8, warmup_steps=500, weight_decay=0.01):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=self.logging_dir,
            logging_steps=10,
            report_to=self.report_to,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        return trainer
