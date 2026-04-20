import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------
# LOAD MODEL
# -----------------------
tokenizer = AutoTokenizer.from_pretrained("model/sms_transformer_model")
model = AutoModelForSequenceClassification.from_pretrained("model/sms_transformer_model")


# -----------------------
# LOAD TEST DATA
# -----------------------
data = pd.read_csv("dataset/test_cases.csv")


# -----------------------
# FRAUD KEYWORDS
# -----------------------
fraud_keywords = [
    "bank", "account", "kyc", "blocked", "verify", "update",
    "click", "urgent", "login", "password", "otp"
]


predictions = []
actuals = []


# -----------------------
# PREDICTION LOOP
# -----------------------
for _, row in data.iterrows():

    text = row["message"]
    label = row["label"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)

    prob = probs[0][1].item()

    # -----------------------
    # KEYWORD CHECK
    # -----------------------
    text_lower = text.lower()
    keyword_flag = any(word in text_lower for word in fraud_keywords)

    # -----------------------
    # FINAL PREDICTION (IMPROVED)
    # -----------------------
    pred = 1 if prob > 0.3 or keyword_flag else 0

    predictions.append(pred)
    actuals.append(label)


# -----------------------
# METRICS
# -----------------------
print("\n📊 MODEL PERFORMANCE")

print("Accuracy :", accuracy_score(actuals, predictions))
print("Precision:", precision_score(actuals, predictions))
print("Recall   :", recall_score(actuals, predictions))
print("F1 Score :", f1_score(actuals, predictions))