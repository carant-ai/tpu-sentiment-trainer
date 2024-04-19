import torch
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import pandas as pd
from math import ceil
from datetime import datetime

# 1. Set up PyTorch/XLA for TPU usage
device = xm.xla_device()

# 2. Load the pre-trained model
model_name = "mdhugol/indonesia-bert-sentiment-classification" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
label_index = {'positive': 0, 'neutral': 1, 'negative': 2}
model.to(device)
model_wrapper = {'model': model, 'tokenizer': tokenizer, 'label_index': label_index}

# 3. Data Preparation
dataset = load_dataset("thonyyy/indonesian_sentiment_dataset", split = 'train').to_pandas()

# 4. Inference
def inference_pipeline(input_text, device, model, tokenizer, label_index):
    reverse_label_index = {int(v): k for k,v in label_index.items()}
    tokenized_text = tokenizer(
        input_text,
        return_tensors = 'pt',
        padding = "max_length",
        truncation = True,
        max_length = 128
        ).to(device)
    with torch.no_grad():
        score = np.array(model(**tokenized_text).logits.to('cpu').softmax(axis = 1))
    del tokenized_text
    result = {x:score.T[label_index[x]] for x in reverse_label_index.values()}
    labels = [reverse_label_index[i] for i in np.argmax(score,axis=1)]
    result['predicted_label'] = labels
    return result

def dataframe_inference(df, device, model_wrapper, infer_batch_size):
    list_label = list(model_wrapper['label_index'].keys())
    frame = []
    num_batch = ceil(len(df)/infer_batch_size)
    for idx in range(num_batch):
        temp  = df.iloc[idx*infer_batch_size:(1+idx)*infer_batch_size]
        result = inference_pipeline(list(temp['text']), device, **model_wrapper)
        for col in list_label+['predicted_label']:
            assert len(temp) == len(result[col])
            col_name = col + '_score'*(col != "predicted_label")
            temp[col_name] = result[col]
        frame.append(temp)
    df = pd.concat(frame)
    return df

t = datetime.now()
df = dataframe_inference(dataset, device, model_wrapper, 512)
print(datetime.now()-t)

# 5. Retrieve Results
df.to_csv("result.csv")