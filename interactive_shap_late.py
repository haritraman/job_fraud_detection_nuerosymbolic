import torch
import torch.nn as nn
import re
import shap
import numpy as np
import warnings
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertTokenizer

warnings.filterwarnings("ignore")

# ================= 1. DEFINE THE LATE FUSION ARCHITECTURE =================
class NeurosymbolicClassifier(nn.Module):
    def __init__(self, rule_dim=15): # Matches your 15 rules
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        
        # Tower 1: Text Logic
        self.text_classifier = nn.Linear(768, 1)
        # Tower 2: Symbolic Logic
        self.rule_classifier = nn.Linear(rule_dim, 1)

    def forward(self, input_ids, attention_mask, rules):
        # 1. Text Analysis 
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        text_logits = self.text_classifier(cls)
        
        # 2. Rule Analysis
        rule_logits = self.rule_classifier(rules)
        
        # 3. EXPLICIT LATE FUSION (Logit Summation)
        # The AI combines both opinions mathematically here
        final_logits = text_logits + rule_logits
        
        return final_logits

# ================= 2. DEFINE THE 15 RULES ENGINE =================
def apply_symbolic_rules(row):
    desc = str(row.get('description', '')).lower()
    title = str(row.get('title', '')).lower()
    raw_title = str(row.get('title', ''))
    profile = str(row.get('company_profile', '')).lower()
    try:
        logo = int(row.get('has_company_logo', 0))
    except:
        logo = 0

    vec = [0] * 15
    
    # Original 10 Rules
    if re.search(r'(registration fee|processing fee|security deposit|ssn|social security|bank account|purchase|security fee)', desc): vec[0] = 1
    if re.search(r'(whatsapp|telegram|@gmail|@yahoo|@hotmail|@outlook)', desc): vec[1] = 1
    if re.search(r'(urgent hiring|apply immediately|limited seats|hired on the spot|no interview|start immediately)', desc): vec[2] = 1
    if re.search(r'(no experience|entry level)', desc) and re.search(r'(\$5000|\$5,000|high salary|easy work|1000/week)', desc): vec[3] = 1
    if (raw_title.isupper() and len(raw_title) > 4) or re.search(r'(earn|weekly|quick money|immediate|hiring now)', title): vec[4] = 1
    if re.search(r'(data entry|virtual assistant|package handler|reshipping)', title): vec[5] = 1
    if len(profile) < 50: vec[6] = 1
    if re.search(r'(fast-growing|leading organization|multinational)', profile) and not re.search(r'(www\.|http|\.com|\.org|\.net)', profile): vec[7] = 1
    if logo == 0 and vec[6] == 1: vec[8] = 1
    if logo == 0 and vec[1] == 1: vec[9] = 1
    
    # The 5 New Modern Rules
    if re.search(r'(usdt|crypto wallet|bitcoin|ethereum|metamask|trust wallet)', desc): vec[10] = 1
    if re.search(r'(bit\.ly|tinyurl\.com|google forms|forms\.gle|linktr\.ee)', desc): vec[11] = 1
    if re.search(r'(download our app|rate 5 stars|app testing|click tasks|daily tasks)', desc): vec[12] = 1
    if re.search(r'(must be 18|basic english|internet connection|smartphone required)', desc): vec[13] = 1
    if desc.count('!') > 4: vec[14] = 1
    
    return vec

# ================= 3. SETUP & LOAD MODEL =================
print("Loading model and tokenizer... (This takes a few seconds)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the Late Fusion model
model = NeurosymbolicClassifier(rule_dim=15).to(device)
model.load_state_dict(torch.load("emscad_late_neurosymbolic_shap.pth", map_location=device))
model.eval()

# ================= 4. SHAP WRAPPER =================
def predict_for_shap(texts):
    if isinstance(texts, str):
        texts = [texts]
    else:
        texts = list(texts)
        
    probs = []
    for text in texts:
        text_str = str(text)
        dummy_row = {'description': text_str, 'title': text_str, 'company_profile': text_str, 'has_company_logo': 0}
        rule_vector = apply_symbolic_rules(dummy_row)
        enc = tokenizer(text_str, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        rules_tensor = torch.tensor([rule_vector], dtype=torch.float).to(device)
        
        with torch.no_grad():
            output = model(input_ids, attention_mask, rules_tensor)
            prob = torch.sigmoid(output).item()
        probs.append(prob)
        
    return np.array(probs).flatten() 

# ================= 5. INTERACTIVE TERMINAL LOOP =================
if __name__ == "__main__":
    print("\n✅ System Ready!")
    print("Type 'exit' or 'quit' to stop the program.\n")
    
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_for_shap, masker)

    RULE_NAMES = [
        "Financial/PII Request (Fee, Security Deposit, SSN)",
        "Suspicious External Comms (WhatsApp, Telegram, Personal Email)",
        "High Pressure / Urgency Tactics",
        "Unrealistic Offer (Entry-level + High Salary)",
        "Clickbait / Scam Title",
        "High-Risk Scam Role (Data Entry, Reshipping)",
        "Ghost Company Profile (Too Short)",
        "Vague Company Profile (No corporate web links)",
        "Low Effort Setup",
        "Phishing Funnel",
        "Crypto / Web3 Payment Request",
        "Sketchy URL / Link Shortener",
        "Click Farm / App Testing Tasks",
        "Zero-Skill Requirement (Must have smartphone, etc.)",
        "Spammy Formatting (Excessive Exclamation Marks)"
    ]

    while True:
        user_input = input("\n📝 Paste a Job Description to analyze (or type 'exit'):\n> ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down...")
            break
            
        if not user_input.strip():
            continue
            
        print("\n🔍 Analyzing text... (Instant Prediction)")
        
        # --- 1. INSTANT NEUROSYMBOLIC PREDICTION ---
        dummy_row = {'description': user_input, 'title': user_input, 'company_profile': user_input, 'has_company_logo': 0}
        rule_vector = apply_symbolic_rules(dummy_row)
        
        enc = tokenizer(user_input, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        rules_tensor = torch.tensor([rule_vector], dtype=torch.float).to(device)
        
        with torch.no_grad():
            output = model(input_ids, attention_mask, rules_tensor)
            prob = torch.sigmoid(output).item()
            
        # Pure Mathematical Decision (No Hard Override!)
        is_fake = prob > 0.5
        confidence = prob * 100 if is_fake else (1 - prob) * 100
        alert_msg = "🚨🚨 SYSTEM ALERT: FAKE JOB DETECTED 🚨🚨" if is_fake else "✅✅ CLEAR: LEGITIMATE JOB ✅✅"
        
        # --- 2. PRINT THE CLEAR USER INTERFACE ---
        print("\n" + "="*60)
        print(alert_msg)
        print(f"🧠 Late Fusion Neural Score: {prob * 100:.2f}% scam probability")
        print("="*60)
        
        triggered_rules = [RULE_NAMES[i] for i, val in enumerate(rule_vector) if val == 1]
        
        print("\n📌 LOGICAL RULE BREAKDOWN:")
        if triggered_rules:
            for rule in triggered_rules:
                print(f"  🚩 {rule}")
        else:
            print("  ✅ No suspicious red flags detected in the text formatting.")
            
        # --- 3. ON-DEMAND SHAP EXPLANATION ---
        want_shap = input("\nDo you want to generate a visual SHAP explanation? (y/n): ")
        
        if want_shap.lower() in ['y', 'yes']:
            print("\n📊 Generating deep learning visual proof (SHAP)... (Please wait ~30 seconds)")
            shap_values = explainer([user_input])
            
            plt.figure(figsize=(10, 6))
            chart_title = "FAKE JOB SIGNALS" if is_fake else "LEGITIMATE JOB SIGNALS"
            plt.title(f"Explainable AI: {chart_title}\n(Red = Pushes towards Scam, Blue = Pushes towards Legit)", fontsize=12, pad=20)
            
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            plt.tight_layout()
            plt.show()
            
            print("💡 Image generated! Close the image window when you are ready to test another job.")
        else:
            print("⏭️ Skipping visual explanation.")