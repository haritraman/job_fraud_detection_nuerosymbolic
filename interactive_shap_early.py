import torch
import torch.nn as nn
import re
import shap
import numpy as np
import warnings
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertTokenizer

warnings.filterwarnings("ignore")

# ================= 1. DEFINE THE MODEL ARCHITECTURE =================
class NeurosymbolicClassifier(nn.Module):
    def __init__(self, rule_dim=15): # UPDATED TO 15
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768 + rule_dim, 1)

    def forward(self, input_ids, attention_mask, rules):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        combined = torch.cat((cls, rules), dim=1)
        return self.classifier(combined)

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

    vec = [0] * 15 # UPDATED TO 15
    
    # Original 10 Rules
    if re.search(r'(registration fee|processing fee|security deposit|ssn|social security|bank account)', desc): vec[0] = 1
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

# Make sure to load the correct Early Fusion model file!
model = NeurosymbolicClassifier(rule_dim=15).to(device)
# Replace with your actual 15-rule Early Fusion .pth filename if it's different
model.load_state_dict(torch.load("emscad_early_neurosymbolic_shap.pth", map_location=device))
model.eval()

# ================= 4. SHAP WRAPPER (BUG-FIXED FOR VS CODE) =================
def predict_for_shap(texts):
    # Safely handle SHAP's NumPy array inputs
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
        
    return np.array(probs).flatten() # Added flatten() to prevent crashes

# ================= 5. INTERACTIVE TERMINAL LOOP =================
if __name__ == "__main__":
    print("\n✅ System Ready!")
    print("Type 'exit' or 'quit' to stop the program.\n")
    
    # Setup SHAP explainer once to save time
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_for_shap, masker)

    # 15 Human-readable names
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
            
        # Fatal Override Check
        is_fatal = rule_vector[0] == 1 or rule_vector[1] == 1
        
        if is_fatal:
            is_fake = True
            confidence = 100.0
            alert_msg = "🚨🚨 SYSTEM ALERT: FATAL SCAM OVERRIDE 🚨🚨"
        else:
            is_fake = prob > 0.5
            confidence = prob * 100 if is_fake else (1 - prob) * 100
            alert_msg = "🚨🚨 SYSTEM ALERT: FAKE JOB DETECTED 🚨🚨" if is_fake else "✅✅ CLEAR: LEGITIMATE JOB ✅✅"
        
        # --- 2. PRINT THE CLEAR USER INTERFACE ---
        print("\n" + "="*60)
        print(alert_msg)
        
        # NEW: Print the exact probability the neural network calculated
        print(f"🧠 Neural Network Raw Score: {prob * 100:.2f}% scam probability")
        
        # Print the final system decision
        if is_fatal:
            print("🛡️ Final System Decision: 100.00% FAKE (Hard Override Triggered)")
        else:
            print(f"🛡️ Final System Decision: {confidence:.2f}% {'FAKE' if is_fake else 'LEGITIMATE'}")
            
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