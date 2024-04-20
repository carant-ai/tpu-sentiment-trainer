import os
import dotenv
if '.env' in os.listdir():
    dotenv.load_dotenv('.env')
import uuid
from datetime import datetime

import pandas as pd
import numpy as np

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import torch_xla.core.xla_model as xm

if __name__ == "__main__":
    start_main = datetime.now()

    CPU_COUNT = 240
    show_stats = False
    print("Read model hyperparameter from .env")
    model_id = str(uuid.uuid4())
    base_model_name = os.getenv("BASE_MODEL_NAME")
    learning_rate = float(os.getenv('LEARNING_RATE'))
    batch_size = int(os.getenv('BATCH_SIZE'))
    epoch = int(os.getenv('EPOCH'))
    val_size = float(os.getenv('VAL_SIZE'))
    id_dataset_name =  os.getenv("ID_DATASET")
    en_dataset_name =  os.getenv("EN_DATASET")

    print("Loading dataset from huggingface hub")
    id_dataset = load_dataset(id_dataset_name, split = 'train').select(range(1000))
    en_dataset = load_dataset(en_dataset_name, split = 'train').select(range(1000))
    if show_stats:
        print("Dataset distribution")
        print("Indonesian rows:", len(id_dataset))
        print("Label distribution")
        print(pd.Series(id_dataset["label_text"]).value_counts())
        print("Source distribution")
        print(pd.Series(id_dataset["source"]).value_counts())
        print("English rows:", len(en_dataset))
        print("Label distribution")
        print(pd.Series(en_dataset["label_text"]).value_counts())
        print("Source distribution")
        print(pd.Series(en_dataset["source"]).value_counts())

    dataset = concatenate_datasets([id_dataset, en_dataset]).shuffle(seed = 42)
    dataset = dataset.filter(lambda example: example['label_text'] != 'unlabeled', num_proc = CPU_COUNT)
    col_names = dataset.column_names
    num_rows = len(dataset)
    
    label_index = {
        'negative' : 0,
        'neutral' : 1, 
        'positive' : 2
        }
    reverse_label_index = {str(v):k for k,v in label_index.items()}

    def convert_labels_to_ids(examples):
        examples['label'] = [label_index[x] for x in examples['label_text']]
        return examples

    print('Convert string labels into integer labels')
    dataset = dataset.map(convert_labels_to_ids, batched = True, num_proc = CPU_COUNT)

    print('Loading model to finetune')
    device = xm.xla_device()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels = len(reverse_label_index),
        output_attentions = False,
        output_hidden_states = False,
        label2id = label_index,
        id2label = reverse_label_index
        ).to(device)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding = "max_length",
            truncation = True,
            max_length = 128)

    print(f'Take {np.round(val_size*100,2)} percent data as evaluation dataset')
    train_dataset = dataset.select(range(int(num_rows*val_size), num_rows))
    train_dataset = train_dataset.map(tokenize_function, batched = True, remove_columns = col_names, num_proc = CPU_COUNT)
    eval_dataset = dataset.select(range(int(num_rows*val_size)))
    eval_dataset = eval_dataset.map(tokenize_function, batched = True, remove_columns = col_names, num_proc = CPU_COUNT)

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    output_dir = f"model/{model_id}"
    training_args = TrainingArguments(output_dir=f"./model_logs/{model_id}",
                                    evaluation_strategy = "epoch",
                                    per_device_eval_batch_size = batch_size,
                                    per_device_train_batch_size = batch_size,
                                    num_train_epochs = epoch,
                                    save_strategy = "no",
                                    learning_rate = learning_rate,
                                    tpu_num_cores = 4
                                    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        compute_metrics = compute_metrics,
    )
    
    print('Start training model (all layers)')
    started_at = datetime.now()
    trainer.train()
    finished_at = datetime.now()

    print(f'Saving model to {output_dir}')
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print('Predicting label on evaluation dataset')
    validation_metadata = trainer.evaluate()

    model_metadata = {'model_id': model_id,
                      'base_model_name' : base_model_name,
                      'num_layers' : num_layers,
                      'epoch' : epoch,
                      'learning_rate' : learning_rate,
                      'batch_size' : batch_size,
                      'val_accuracy' : validation_metadata['eval_accuracy'],
                      'val_loss' : validation_metadata['eval_loss'],
                      'started_at' : started_at,
                      'finished_at' : finished_at,
                      'total_time' : (finished_at - started_at).total_seconds() 
                      }
    print(model_metadata)