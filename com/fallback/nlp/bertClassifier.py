from transformers import AutoTokenizer, BertForSequenceClassification
import torch
import numpy as np
import os

model_path = os.environ['MODEL_PATH']


class BertClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def classify(self, sent):
        encoded_dict = self.tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=70,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_id = torch.cat([encoded_dict['input_ids']], dim=0)
        attention_mask = torch.cat([encoded_dict['attention_mask']], dim=0)
        with torch.no_grad():
            outputs = self.model(input_id, token_type_ids=None, attention_mask=attention_mask)
        label = np.argmax(outputs[0], axis=1).flatten()
        return label.item()


def check_correctness(sent):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    # print(model)
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=70,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    # print(encoded_dict)
    input_id = torch.cat([encoded_dict['input_ids']], dim=0)
    attention_mask = torch.cat([encoded_dict['attention_mask']], dim=0)
    # print(input_id, attention_mask)

    model.eval()
    with torch.no_grad():
        outputs = model(input_id, token_type_ids=None,
                        attention_mask=attention_mask)
    logits = outputs[0]
    # print(logits)
    label = np.argmax(logits, axis=1).flatten()
    return label.item()


# check_correctness('Я был в Астане. Купил там атвлулу.')