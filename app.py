from flask import Flask, request, jsonify, send_from_directory
import torch, re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__, static_folder=".")

sms_tok   = AutoTokenizer.from_pretrained("model/sms_transformer_model")
sms_mod   = AutoModelForSequenceClassification.from_pretrained("model/sms_transformer_model")
email_tok = AutoTokenizer.from_pretrained("model/email_transformer_model")
email_mod = AutoModelForSequenceClassification.from_pretrained("model/email_transformer_model")

sms_mod.eval()
email_mod.eval()

# Keywords that ONLY matter in suspicious context
fraud_keywords = [
    "kyc", "blocked", "suspended", "verify your account",
    "click here", "login now", "confirm your",
    "prize", "lottery", "winner", "reward", "free cash",
    "earn money", "investment scheme",
    "enter otp", "share otp", "share password",
    "your account will be", "will be blocked", "will be suspended"
]

# Trusted senders — skip fraud check
trusted_senders = [
    "jio", "airtel", "bsnl", "vi ", "vodafone",
    "sbi", "hdfc", "icici", "axis", "kotak",
    "amazon", "flipkart", "swiggy", "zomato",
    "team jio", "team airtel", "missed call"
]

def detect_url(text):
    return bool(re.search(r"http[s]?://|www\.", text))

def is_trusted(text):
    tl = text.lower()
    return any(t in tl for t in trusted_senders)

@app.route("/")
def index():
    return send_from_directory(".", "fraud_detector.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data  = request.json
    msg   = data.get("message", "")
    typ   = data.get("type", "SMS")

    tok = sms_tok   if typ == "SMS" else email_tok
    mod = sms_mod   if typ == "SMS" else email_mod

    inputs = tok(msg, return_tensors="pt", truncation=True, padding=True)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = mod(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    fp    = float(probs[0][1])

    ml      = msg.lower()
    trusted = is_trusted(msg)

    # Phrase-based keywords (more precise than single words)
    kwH  = any(k in ml for k in fraud_keywords)
    urlH = detect_url(msg)
    urgH = any(w in ml for w in ["urgent", "immediately", "act now", "last chance"])

    # Risk scoring — weighted properly
    rs = fp * 100

    if kwH:  rs += 20
    if urlH: rs += 20
    if urgH: rs += 10

    # Reduce score for trusted senders
    if trusted: rs = rs * 0.3

    rs = min(round(rs), 100)

    # Final verdict — stricter threshold
    # AI must be >0.75 confident OR multiple signals together
    signals = sum([kwH, urlH, urgH])
    fraud = (
        (fp > 0.75) or                      # AI very confident
        (fp > 0.5 and signals >= 2) or       # AI moderately confident + multiple signals
        (kwH and urlH)                       # keyword + URL together
    ) and not trusted                        # never flag trusted senders

    return jsonify({
        "fraud": bool(fraud),
        "fp":    round(fp * 100),
        "rs":    rs,
        "kwH":   bool(kwH),
        "urlH":  bool(urlH),
        "urgH":  bool(urgH),
        "fpRaw": bool(fp > 0.5),
        "trusted": bool(trusted)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)