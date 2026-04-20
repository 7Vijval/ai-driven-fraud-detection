import joblib
import re


# TEXT CLEANING
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text


# FRAUD KEYWORDS
fraud_keywords = [
    "win", "winner", "prize", "lottery", "urgent",
    "claim", "reward", "free", "selected",
    "click", "offer", "cash", "gift"
]


# LOAD MODEL
model = joblib.load("model/sms_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

print("SMS Fraud Detection System Ready")


while True:

    sms = input("\nEnter SMS message: ")

    cleaned_sms = clean_text(sms)

    sms_vector = vectorizer.transform([cleaned_sms])

    prediction = model.predict(sms_vector)[0]

    # RULE-BASED CHECK
    keyword_flag = any(word in cleaned_sms for word in fraud_keywords)

    if prediction == 1 or keyword_flag:
        print("⚠ FRAUDULENT SMS DETECTED")
    else:
        print("SAFE SMS")