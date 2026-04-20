import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------
# LOAD MODEL
# -----------------------
tokenizer = AutoTokenizer.from_pretrained("model/sms_transformer_model")
model = AutoModelForSequenceClassification.from_pretrained("model/sms_transformer_model")
model.eval()

print("AI SMS Fraud Detection Ready")

# -----------------------
# PREDICTION LOOP
# -----------------------
while True:
    print("\nPaste your full SMS (press ENTER twice to finish):")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    sms = " ".join(lines)

    if sms.strip() == "":
        print("No input detected. Try again.")
        continue

    inputs = tokenizer(sms, return_tensors="pt", truncation=True, padding=True)
    inputs.pop("token_type_ids", None)   # DistilBERT fix

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    fraud_prob = probs[0][1].item()

    if fraud_prob > 0.5:
        print("\nFRAUD DETECTED")
    else:
        print("\nSAFE MESSAGE")

    print("Fraud probability:", round(fraud_prob * 100, 2), "%")