from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
import torch.nn as nn

class LLMModelBuilder:
    def __init__(self, model_name: str, num_labels: int = None):
        self.model_name = model_name
        self.num_labels = num_labels

    def build_model(self):
        if self.num_labels is not None:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return model
