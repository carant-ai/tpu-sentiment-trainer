from lightning import LightningModule, LightningDataModule, Trainer
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from torch import nn, optim
import torch
import os

def collator(batch):
    keys = batch[0].keys()
    new_batch = {}

    for key in keys:
        if key == "labels":
            labels = torch.Tensor([(example[key]) for example in batch])
        else:
            new_batch[key] = torch.cat([(example[key].view(1, -1)) for example in batch], dim = 0)

    return new_batch, labels



def tokenize(dataset, tokenizer, **kwargs):
    mapping = {"negative": 0, "neutral": 1, "positive": 2}

    def batch_tokenize(batch):
        token = tokenizer(
            batch["text"],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        token["labels"] = [mapping[label] for label in batch["label_text"]]
        return token

    return dataset.map(
        batch_tokenize,
        num_proc=16,
        batched=True,
        remove_columns=["text", "label_text", "source", "split"],
    )


class Model(LightningModule):
    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self.model(**features).logits
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_epoch = True, on_step = True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        return optimizer


class Sentence(LightningDataModule):
    def __init__(self, model_name: str = "xlm-roberta-base", val_ratio=0.1):
        super().__init__()
        access_token = os.getenv("ACCESS_TOKEN")
        id_dataset = load_dataset(
            "thonyyy/indonesian_sentiment_dataset_v1", split="train", token=access_token
        )
        en_dataset = load_dataset(
            "thonyyy/english_sentiment_dataset_v1", split="train", token=access_token
        )
        concat_ds = concatenate_datasets([id_dataset, en_dataset])
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_ds = tokenize(concat_ds, tokenizer)
        self.train_ds.set_format("torch", columns=["input_ids", "attention_mask"])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(
        self,
    ):
        return DataLoader(self.train_ds, batch_size=512, collate_fn = collator, num_workers = 16)


def main_2():
    data = Sentence()
    loader = DataLoader(data.train_ds, batch_size = 4, collate_fn = collator)
    for batch in loader:
        print(batch)
        break

def main():
    model = Model()
    data = Sentence()
    trainer = Trainer(accelerator = "cpu")
    trainer.fit(model, data)


if __name__ == "__main__":
    main()