import pandas as pd
import re
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
# LOAD EMAIL DATASET
# -----------------------
data = pd.read_csv("dataset/email_spam.csv")
data = data[['label', 'text']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
data['message'] = data['message'].apply(clean_text)
data = data.dropna()

print(f"Email dataset loaded: {len(data)} samples")
print(f"Ham: {(data.label==0).sum()} | Spam: {(data.label==1).sum()}")

# -----------------------
# TRAIN / VAL / TEST SPLIT
# -----------------------
train_df, temp_df = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

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
        max_length=256        # emails are longer than SMS
    )
    enc.pop('token_type_ids', None)   # DistilBERT fix
    return enc

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))
test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(['message'])
val_dataset   = val_dataset.remove_columns(['message'])
test_dataset  = test_dataset.remove_columns(['message'])

train_dataset.set_format('torch')
val_dataset.set_format('torch')
test_dataset.set_format('torch')

# -----------------------
# CLASS WEIGHTS
# -----------------------
n_ham  = (train_df.label == 0).sum()
n_spam = (train_df.label == 1).sum()
total  = len(train_df)
class_weights = torch.tensor([total / (2 * n_ham), total / (2 * n_spam)], dtype=torch.float)
print(f"Class weights — Ham: {class_weights[0]:.3f} | Spam: {class_weights[1]:.3f}")

# -----------------------
# WEIGHTED TRAINER
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
    output_dir='model/email_results',
    num_train_epochs=5,
    per_device_train_batch_size=16,    # smaller batch — emails are longer
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir='model/email_logs',
    logging_steps=50,
    fp16=torch.cuda.is_available(),
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

print("\nStarting email model training...")
trainer.train()

# -----------------------
# EVALUATE
# -----------------------
print("\nFinal evaluation on test set:")
results = trainer.evaluate(test_dataset)
print(results)

# -----------------------
# SAVE
# -----------------------
trainer.save_model('model/email_transformer_model')
tokenizer.save_pretrained('model/email_transformer_model')
print("\nEmail Transformer Model Saved Successfully")