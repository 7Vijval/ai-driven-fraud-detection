import pandas as pd
import re
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

MODEL_NAME = "distilbert-base-uncased"

# -----------------------
# TEXT CLEANING
# -----------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------
# LOAD DATASET
# -----------------------
data = pd.read_csv("dataset/spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
data['message'] = data['message'].apply(clean_text)
data = data.dropna()

print(f"Dataset loaded: {len(data)} samples")
print(f"Ham: {(data.label==0).sum()} | Spam: {(data.label==1).sum()}")

# -----------------------
# TRAIN / VAL / TEST SPLIT
# -----------------------
train_df, temp_df = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# -----------------------
# TOKENIZER
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    enc = tokenizer(
        example['message'],
        padding='max_length',
        truncation=True,
        max_length=128
    )
    # Remove token_type_ids — DistilBERT doesn't use them
    enc.pop('token_type_ids', None)
    return enc

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))
test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

for ds in [train_dataset, val_dataset, test_dataset]:
    for col in ['message']:
        if col in ds.column_names:
            ds = ds.remove_columns([col])

train_dataset = train_dataset.remove_columns(['message'])
val_dataset   = val_dataset.remove_columns(['message'])
test_dataset  = test_dataset.remove_columns(['message'])

train_dataset.set_format('torch')
val_dataset.set_format('torch')
test_dataset.set_format('torch')

# -----------------------
# CLASS WEIGHTS (handles imbalance)
# -----------------------
n_ham  = (train_df.label == 0).sum()
n_spam = (train_df.label == 1).sum()
total  = len(train_df)
class_weights = torch.tensor([total / (2 * n_ham), total / (2 * n_spam)], dtype=torch.float)
print(f"Class weights — Ham: {class_weights[0]:.3f} | Spam: {class_weights[1]:.3f}")

# -----------------------
# MODEL WITH WEIGHTED LOSS
# -----------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# -----------------------
# METRICS
# -----------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': round(acc,4), 'precision': round(precision,4),
            'recall': round(recall,4), 'f1': round(f1,4)}

# -----------------------
# TRAINING ARGS
# -----------------------
training_args = TrainingArguments(
    output_dir='model/results',
    num_train_epochs=5,               # was 1 — needs more
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,                # regularization
    warmup_ratio=0.1,                 # gradual warmup
    lr_scheduler_type='cosine',       # cosine decay
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir='model/logs',
    logging_steps=50,
    fp16=torch.cuda.is_available(),   # faster on GPU
    seed=42,
)

# -----------------------
# TRAIN
# -----------------------
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\nStarting training...")
trainer.train()

# -----------------------
# EVALUATE ON TEST SET
# -----------------------
print("\nFinal evaluation on test set:")
results = trainer.evaluate(test_dataset)
print(results)

# -----------------------
# SAVE
# -----------------------
trainer.save_model('model/sms_transformer_model')
tokenizer.save_pretrained('model/sms_transformer_model')
print("\nSMS Transformer Model Saved Successfully")