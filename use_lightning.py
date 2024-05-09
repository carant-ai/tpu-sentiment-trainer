from lightning import LightningModule, LightningDataModule
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

def tokenize(dataset, tokenizer, label_index, **kwargs):
    def batch_tokenize(batch):
        result = tokenizer(batch['text'], max_length = 128, padding = "max_legnth", truncation = True)
        result['label'] = [label_index[x] for x in examples['label_text']]
        return result
    return dataset.map(batch_tokenize, num_proc = 16, batched = True)
