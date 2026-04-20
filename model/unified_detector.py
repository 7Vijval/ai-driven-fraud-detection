import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------
# LOAD MODELS
# -----------------------
sms_tokenizer   = AutoTokenizer.from_pretrained("model/sms_transformer_model")
sms_model       = AutoModelForSequenceClassification.from_pretrained("model/sms_transformer_model")
email_tokenizer = AutoTokenizer.from_pretrained("model/email_transformer_model")
email_model     = AutoModelForSequenceClassification.from_pretrained("model/email_transformer_model")

sms_model.eval()
email_model.eval()

# -----------------------
# FRAUD KEYWORDS
# -----------------------
fraud_keywords = [
    "bank", "account", "kyc", "blocked", "suspended", "verify", "update",
    "click", "link", "login", "confirm", "secure",
    "urgent", "immediately", "now", "alert",
    "prize", "lottery", "winner", "reward", "cash",
    "earn", "income", "investment",
    "password", "otp", "pin"
]

urgency_words = ["urgent", "immediately", "now", "alert"]

def detect_url(text):
    return bool(re.search(r"http[s]?://|www\.", text))

print("Unified Fraud Detection System Ready")

# -----------------------
# MAIN LOOP
# -----------------------
while True:
    print("\nSelect Input Type:")
    print("1. SMS")
    print("2. Email")

    choice = input("Enter choice (1/2): ").strip()

    if choice not in ["1", "2"]:
        print("Invalid choice. Try again.")
        continue

    print("\nPaste your message (press ENTER twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    message = " ".join(lines)

    if message.strip() == "":
        print("No input detected.")
        continue

    # -----------------------
    # SELECT MODEL
    # -----------------------
    if choice == "1":
        tokenizer = sms_tokenizer
        model     = sms_model
        msg_type  = "SMS"
    else:
        tokenizer = email_tokenizer
        model     = email_model
        msg_type  = "Email"

    # -----------------------
    # MODEL PREDICTION
    # -----------------------
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True)
    inputs.pop("token_type_ids", None)   # DistilBERT fix

    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.softmax(outputs.logits, dim=1)
    fraud_prob = probs[0][1].item()

    # -----------------------
    # KEYWORD & URL CHECK
    # -----------------------
    message_lower   = message.lower()
    keyword_matches = [w for w in fraud_keywords if w in message_lower]
    keyword_flag    = len(keyword_matches) >= 2
    url_flag        = detect_url(message)
    urgency_flag    = any(w in message_lower for w in urgency_words)

    # -----------------------
    # RISK SCORING
    # -----------------------
    risk_score = fraud_prob * 100
    if keyword_flag:  risk_score += 20
    if url_flag:      risk_score += 25
    if urgency_flag:  risk_score += 15
    risk_score = min(risk_score, 100)

    risk_level = "LOW" if risk_score < 30 else "MEDIUM" if risk_score < 70 else "HIGH"

    # -----------------------
    # RESULT
    # -----------------------
    print(f"\nMessage Type : {msg_type}")
    print(f"Verdict      : {'FRAUD DETECTED' if (fraud_prob > 0.5 or keyword_flag or url_flag) else 'SAFE MESSAGE'}")
    print(f"Risk Score   : {round(risk_score, 2)}%")
    print(f"Risk Level   : {risk_level}")
    print(f"AI Confidence: {round(fraud_prob * 100, 2)}%")

    reasons = []
    if keyword_flag: reasons.append(f"phishing keywords ({', '.join(keyword_matches)})")
    if url_flag:     reasons.append("suspicious URL")
    if urgency_flag: reasons.append("urgency language")
    if reasons:
        print(f"Reasons      : {', '.join(reasons)}")

    # -----------------------
    # SAVE
    # -----------------------
    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(f"{msg_type} | {round(risk_score,2)}% | {risk_level} | {message[:80]}\n")