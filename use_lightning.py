from lightning import LightningModule, LightningDataModule, Trainer
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from torch import nn, optim
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import os
import dotenv
if '.env' in os.listdir():
    dotenv.load_dotenv('.env')

def collator(batch):
    keys = batch[0].keys()
    new_batch = {}
    for key in keys:
        if key == "labels":
            labels = torch.cat([(example[key].view(1, -1)) for example in batch], dim = 0).view(-1)
        else:
            new_batch[key] = torch.cat([(example[key].view(1, -1)) for example in batch], dim = 0)
    return new_batch, labels

def tokenize(dataset, tokenizer, label_index):
    def batch_tokenize(batch):
        token = tokenizer(
            batch["text"],
            max_length = 128,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
        )

        token["labels"] = [label_index[label] for label in batch["label_text"]]
        return token

    return dataset.map(
        batch_tokenize,
        num_proc = 16,
        batched = True,
        remove_columns = ["text", "label_text", "source", "split"],
    )

class Model(LightningModule):
    def __init__(self, model_name: str = "xlm-roberta-base", label_index: dict = {}):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels = len(label_index),
            label2id = label_index,
            id2label = {v:k for k,v in label_index.items()}
        )
        self.save_hyperparameters()
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self.model(**features).logits
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_epoch = True, on_step = True)
        predicted_labels = logits.argmax(axis=1).view(-1)
        accuracy = (predicted_labels == labels).float().mean()
        self.log("train_accuracy", accuracy, on_epoch = True, on_step = True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits = self.model(**features).logits
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss, on_epoch = True, on_step = True)
        predicted_labels = logits.argmax(axis=1).view(-1)
        accuracy = (predicted_labels == labels).float().mean()
        self.log("val_accuracy", accuracy, on_epoch = True, on_step = True)
        return loss

    def configure_optimizers(self):
        weight_decay = 0.01
        param_optimizer = list(self.named_parameters())

        # Remove LayerNorm from weight decay params
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=float(os.getenv("LEARNING_RATE")))
        return optimizer


class Sentence(LightningDataModule):
    def __init__(self, model_name: str = "xlm-roberta-base", val_ratio = 0.05, label_index = {}):
        super().__init__()
        access_token = os.getenv("ACCESS_TOKEN")
        id_dataset = load_dataset(
            os.getenv("ID_DATASET"), split = "train", token = access_token
        )
        en_dataset = load_dataset(
            os.getenv("EN_DATASET"), split = "train", token = access_token
        )
        concat_ds = concatenate_datasets([id_dataset, en_dataset]).shuffle(seed=42)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = tokenize(concat_ds, self.tokenizer, label_index)
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        dataset = dataset.train_test_split(test_size = val_ratio)
        self.train_dataset = dataset['train']
        self.val_dataset = dataset['test']
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = int(os.getenv("BATCH_SIZE")), collate_fn = collator, num_workers = 16)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = int(os.getenv("BATCH_SIZE")), collate_fn = collator, num_workers = 16)

def main():
    label_index = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
        }
    model = Model(model_name = os.getenv("BASE_MODEL_NAME"), label_index = label_index)
    data = Sentence(model_name = os.getenv("BASE_MODEL_NAME"), val_ratio = float(os.getenv("VAL_SIZE")), label_index = label_index)
    checkpoint_callback = ModelCheckpoint(monitor = 'val_loss')
    trainer = Trainer(accelerator = os.getenv("ACCELERATOR"), max_epochs = int(os.getenv("EPOCH")), callbacks =  [checkpoint_callback])
    trainer.fit(model, data)
    m = Model.load_from_checkpoint(checkpoint_callback.best_model_path)
    m.model.push_to_hub(os.getenv("HUB_MODEL_NAME"))
    data.tokenizer.push_to_hub(os.getenv("HUB_MODEL_NAME"))

if __name__ == "__main__":
    main()