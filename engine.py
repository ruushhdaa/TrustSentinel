import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

# =====================
# STEP 1: LOAD DATA
# =====================

import os

print("Loading data...")

LOCAL_FILE = 'train_transaction.csv'
DEMO_FILE = 'demo_transactions.csv'

if os.path.exists(LOCAL_FILE):
    df = pd.read_csv(LOCAL_FILE)
    print("Using full local dataset: train_transaction.csv")
else:
    df = pd.read_csv(DEMO_FILE)
    print("Using demo dataset: demo_transactions.csv")

# =====================
# STEP 2: MORE FEATURES
# =====================

# We now use 40 features instead of 5
# This is what separates us from basic projects

base_features = [
    'TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt',
    'card1', 'card2', 'card3', 'card4', 'card5',
    'addr1', 'addr2', 'dist1', 'dist2',
    'P_emaildomain', 'R_emaildomain',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15'
]

# Only keep columns that exist in the dataset
available = [c for c in base_features if c in df.columns]
df = df[available]
df = df.fillna(0)
df['hour'] = (df['TransactionDT'] / 3600 % 24).astype(int)

# Encode string columns to numbers
le_card4 = LabelEncoder()
le_pemail = LabelEncoder()
le_remail = LabelEncoder()

df['card4_encoded'] = le_card4.fit_transform(df['card4'].astype(str))
df['P_email_encoded'] = le_pemail.fit_transform(df['P_emaildomain'].astype(str))
df['R_email_encoded'] = le_remail.fit_transform(df['R_emaildomain'].astype(str))

print(f"Data ready with {len(df)} transactions and {len(df.columns)} columns")

# =====================
# STEP 3: TRUST PROFILES
# =====================

print("Building trust profiles...")

trust_profiles = df.groupby('card1').agg(
    avg_amount  = ('TransactionAmt', 'mean'),
    max_amount  = ('TransactionAmt', 'max'),
    std_amount  = ('TransactionAmt', 'std'),
    total_txns  = ('TransactionID', 'count'),
    usual_hours = ('hour', lambda x: x.mode()[0] if len(x) > 0 else 12)
).reset_index()

trust_profiles['std_amount'] = trust_profiles['std_amount'].fillna(0)
print(f"Trust profiles built for {len(trust_profiles)} unique cards")

# =====================
# STEP 4: TRAIN WITH SMOTE
# =====================

print("Preparing training data with SMOTE...")

ml_features = [
    'TransactionAmt', 'hour', 'card1', 'card2', 'card3',
    'card4_encoded', 'card5', 'addr1', 'addr2',
    'C1', 'C2', 'C3', 'C4', 'C5',
    'V1', 'V2', 'V3', 'V4', 'V5',
    'V6', 'V7', 'V8', 'V9', 'V10',
    'P_email_encoded', 'R_email_encoded'
]

# Only use features that exist
ml_features = [f for f in ml_features if f in df.columns]

X = df[ml_features]
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Before SMOTE — Fraud cases in training: {y_train.sum()}")

# SMOTE creates synthetic fraud examples so model learns fraud better
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"After SMOTE — Fraud cases in training: {y_train_sm.sum()}")
print("Training Random Forest on balanced data... this will take 3-4 minutes")

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_sm, y_train_sm)

print("Random Forest trained successfully")

# ============================================================
# NEW: Proper metrics calculation
# ============================================================
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    recall_score,
    precision_score
)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# CHANGED: Store as module-level variable for dashboard access
global_model_metrics = {
    "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
    "auc_pr": round(auc(*precision_recall_curve(y_test, y_prob)[:2][::-1]), 4),  # FIXED: correct unpacking
    "f1": round(f1_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "fraud_caught": int(recall_score(y_test, y_pred) * y_test.sum()),
    "total_fraud": int(y_test.sum()),
    "false_positives": int(((y_pred == 1) & (y_test == 0)).sum()),
    "total_blocked": int((y_pred == 1).sum()),
    "threshold": 0.35  # NEW: adjustable threshold
}

print("\n╔══════════════════════════════════════╗")
print("║      MODEL PERFORMANCE METRICS       ║")
print("╠══════════════════════════════════════╣")
print(f"║  AUC-ROC:             {global_model_metrics['auc_roc']:.4f}      ║")
print(f"║  AUC-PR:              {global_model_metrics['auc_pr']:.4f}       ║")
print(f"║  F1 Score:            {global_model_metrics['f1']:.4f}           ║")
print(f"║  Recall (Catch Rate): {global_model_metrics['recall']:.1%}       ║")
print(f"║  Precision:           {global_model_metrics['precision']:.1%}    ║")
print(f"║  Frauds Caught:       {global_model_metrics['fraud_caught']}/{global_model_metrics['total_fraud']}       ║")
print(f"║  False Positives:     {global_model_metrics['false_positives']}          ║")
print(f"║  Total Blocked:       {global_model_metrics['total_blocked']}          ║")
print("╚══════════════════════════════════════╝")

# =====================
# STEP 5: REAL SHAP EXPLAINER
# =====================

print("Setting up SHAP explainer...")

# Use a small sample for SHAP background to keep it fast
background = X_train_sm.sample(100, random_state=42)
explainer = shap.TreeExplainer(rf_model)

print("SHAP explainer ready")

def get_shap_reason(transaction_features):
    try:
        shap_values = explainer.shap_values(
            transaction_features,
            check_additivity=False
        )

        # Shape is (1, 26, 2) — take fraud class at index 1
        shap_array = np.array(shap_values)
        if shap_array.ndim == 3:
            fraud_shap = shap_array[0, :, 1]
        elif shap_array.ndim == 2:
            fraud_shap = shap_array[0]
        else:
            fraud_shap = shap_array.flatten()

        shap_series = pd.Series(
            np.abs(fraud_shap),
            index=ml_features[:len(fraud_shap)]
        )
        top_features = shap_series.nlargest(2).index.tolist()

        reasons = []
        for feat in top_features:
            val = transaction_features[feat].values[0]
            if feat == 'TransactionAmt':
                reasons.append(f"Transaction amount (${round(val)}) was a key fraud signal")
            elif feat == 'hour':
                reasons.append(f"Transaction hour ({int(val)}:00) was suspicious")
            elif feat in ['card1', 'card2', 'card3', 'card4_encoded', 'card5']:
                reasons.append(f"Card profile matched known fraud patterns")
            elif feat in ['addr1', 'addr2']:
                reasons.append(f"Transaction location was anomalous")
            elif feat in ['P_email_encoded', 'R_email_encoded']:
                reasons.append(f"Email domain associated with suspicious activity")
            elif feat.startswith('C'):
                reasons.append(f"Transaction count pattern was unusual")
            elif feat.startswith('V'):
                reasons.append(f"Behavioral fingerprint deviated from norm")
        reasons = list(dict.fromkeys(reasons))    
        return " | ".join(reasons) if reasons else "Multiple features flagged as anomalous"

    except Exception as e:
        return "ML model detected anomalous pattern"

# =====================
# STEP 6: TIER 1 BOUNCER
# =====================

recent_transactions = defaultdict(list)

def tier1_bouncer(transaction, profiles):
    card = transaction['card1']
    amount = transaction['TransactionAmt']
    hour = transaction['hour']
    txn_time = transaction['TransactionDT']

    recent_transactions[card].append(txn_time)
    recent_transactions[card] = [
        t for t in recent_transactions[card]
        if txn_time - t <= 10
    ]
    if len(recent_transactions[card]) >= 4 and amount >= 500:
        return True, "TIER1", "Rapid-fire detected: 4+ transactions in 10 seconds"

    profile = profiles[profiles['card1'] == card]
    if len(profile) > 0:
        avg = profile.iloc[0]['avg_amount']
        if avg > 0 and amount > avg * 15:
            return True, "TIER1", f"Amount is {round(amount/avg)}x higher than card average — immediate block"

    if 2 <= hour <= 4 and amount > 1000:
        return True, "TIER1", f"High value ${round(amount)} at {hour}:00 AM — suspicious timing"

    return False, None, None

# =====================
# STEP 7: RISK SCORE (renamed from trust_score for semantic clarity)
# =====================

def calculate_risk_score(transaction, profiles):  # CHANGED: renamed function
    """
    Returns risk score (0-100) where HIGHER = MORE RISKY
    """
    card = transaction['card1']
    amount = transaction['TransactionAmt']
    hour = transaction['hour']

    profile = profiles[profiles['card1'] == card]

    if len(profile) == 0:
        return 85, "New card with no transaction history"

    profile = profile.iloc[0]
    score = 0
    reasons = []

    if profile['avg_amount'] > 0:
        ratio = amount / profile['avg_amount']
        if ratio > 10:
            score += 50
            reasons.append(f"Amount is {round(ratio)}x higher than card average")
        elif ratio > 5:
            score += 30
            reasons.append(f"Amount is {round(ratio)}x higher than card average")
        elif ratio > 3:
            score += 15
            reasons.append(f"Amount is {round(ratio)}x higher than card average")

    usual_hour = profile['usual_hours']
    if abs(hour - usual_hour) > 8:
        score += 25
        reasons.append(f"Unusual hour ({hour}:00) differs from usual pattern")

    if amount > 5000:
        score += 15
        reasons.append(f"High value transaction: ${round(amount)}")

    score = min(score, 100)
    reason_text = " | ".join(reasons) if reasons else "Transaction matches normal behavior"
    return score, reason_text

# =====================
# STEP 8: TIER 2 ML
# =====================

def tier2_ml(transaction, threshold=None):  # CHANGED: added threshold parameter
    """
    ML-based fraud detection with adjustable threshold
    """
    features = pd.DataFrame([transaction[ml_features].values],
                             columns=ml_features)

    probability = rf_model.predict_proba(features)[0][1]

    # CHANGED: use dashboard-controlled threshold, default 0.5
    if threshold is None:
        threshold = global_model_metrics.get("threshold", 0.5)

    if probability >= threshold:  # CHANGED: use threshold instead of prediction
        shap_reason = get_shap_reason(features)
        reason = f"ML flagged with {round(probability*100)}% confidence — {shap_reason}"
        return True, "TIER2", reason, probability  # CHANGED: return probability

    return False, None, None, probability  # CHANGED: return probability

# =====================
# STEP 9: MODEL HEALTH
# =====================

transaction_history = []
training_avg = X_train['TransactionAmt'].mean()
training_std = X_train['TransactionAmt'].std()

def get_model_health():
    if len(transaction_history) < 10:
        return "HEALTHY", 95

    recent = pd.DataFrame(transaction_history[-50:])
    recent_avg = recent['TransactionAmt'].mean()
    recent_std = recent['TransactionAmt'].std() if len(recent) > 1 else 0

    drift = abs(recent_avg - training_avg) / (training_avg + 1e-9)

    if drift < 0.2:
        return "HEALTHY", max(60, int(95 - drift * 100))
    elif drift < 0.5:
        return "WARNING", max(30, int(70 - drift * 100))
    else:
        return "DRIFT DETECTED", max(5, int(40 - drift * 50))  # CHANGED: better label

# =====================
# STEP 10: FALSE POSITIVE
# =====================

blocked_transactions = []
false_positives = []

def update_false_positive(transaction, verdict, actual_fraud):
    if verdict == "FRAUD":
        blocked_transactions.append(1)
        if actual_fraud == 0:
            false_positives.append(1)

def get_false_positive_rate():
    if len(blocked_transactions) == 0:
        return 0.0
    return round(len(false_positives) / len(blocked_transactions) * 100, 2)

# =====================
# STEP 11: MAIN FUNCTION
# =====================

def analyze_transaction(transaction, threshold=None):  # CHANGED: added threshold parameter
    """
    Main fraud detection pipeline with adjustable ML threshold
    """
    # Add to history for model health
    transaction_history.append({
        'TransactionAmt': transaction['TransactionAmt'],
        'hour': transaction['hour']
    })

    # Tier 1 (rules-based — instant block)
    is_fraud, tier, reason = tier1_bouncer(transaction, trust_profiles)
    if is_fraud:
        risk_score, _ = calculate_risk_score(transaction, trust_profiles)  # CHANGED: renamed
        return {
            "verdict": "FRAUD",
            "reason": reason,
            "risk_score": min(risk_score + 40, 100),  # CHANGED: renamed from trust_score
            "tier": tier,
            "model_health": get_model_health(),
            "false_positive_rate": get_false_positive_rate(),
            "probability": None  # NEW: ML probability field
        }

    # Tier 2 (ML-based detection with threshold)
    is_fraud, tier, reason, probability = tier2_ml(transaction, threshold)  # CHANGED: accept probability
    if is_fraud:
        risk_score, ts_reason = calculate_risk_score(transaction, trust_profiles)  # CHANGED: renamed
        combined = reason
        if ts_reason != "Transaction matches normal behavior":
            combined = reason + " | " + ts_reason
        return {
            "verdict": "FRAUD",
            "reason": combined,
            "risk_score": min(risk_score + 30, 100),  # CHANGED: renamed from trust_score
            "tier": tier,
            "model_health": get_model_health(),
            "false_positive_rate": get_false_positive_rate(),
            "probability": probability  # NEW: include ML confidence
        }

    # Risk score check (behavioral anomaly without ML flag)
    risk_score, reason = calculate_risk_score(transaction, trust_profiles)  # CHANGED: renamed
    if risk_score >= 75:
        return {
            "verdict": "FRAUD",
            "reason": reason,
            "risk_score": risk_score,  # CHANGED: renamed from trust_score
            "tier": "RISK_SCORE",
            "model_health": get_model_health(),
            "false_positive_rate": get_false_positive_rate(),
            "probability": None
        }
    
    # Safe transaction
    if risk_score < 30:
        reason = "Transaction matches normal behavior"
    return {
        "verdict": "SAFE",
        "reason": reason,
        "risk_score": risk_score,  # CHANGED: renamed from trust_score
        "tier": "NONE",
        "model_health": get_model_health(),
        "false_positive_rate": get_false_positive_rate(),
        "probability": probability if 'probability' in locals() else None  # NEW: include even for safe
    }

# =====================
# STEP 12: TEST
# =====================

print("\nTesting on 20 samples (10 fraud + 10 safe):")
print("-" * 70)

test_samples = pd.concat([
    df[df['isFraud'] == 1].head(10),
    df[df['isFraud'] == 0].head(10)
])

correct = 0
total = len(test_samples)

for i, (_, txn) in enumerate(test_samples.iterrows()):
    result = analyze_transaction(txn)
    actual = "FRAUD" if txn['isFraud'] == 1 else "SAFE"
    match = "✓" if result['verdict'] == actual else "✗"
    if result['verdict'] == actual:
        correct += 1
    print(f"{match} Actual={actual} | Predicted={result['verdict']} | "
          f"Score={result['risk_score']} | Tier={result['tier']}")  # CHANGED: risk_score
    print(f"   {result['reason']}")

print(f"\nAccuracy on sample: {correct}/{total}")
print(f"False Positive Rate: {get_false_positive_rate()}%")
print(f"Model Health: {get_model_health()}")

# =====================
# STEP 13: DASHBOARD EXPORTS
# =====================

def set_model_threshold(threshold):
    """Adjust ML fraud detection threshold from dashboard"""
    global_model_metrics["threshold"] = threshold
    print(f"Threshold updated to {threshold}")

def get_model_metrics():
    """Returns all model performance metrics for dashboard"""
    return global_model_metrics.copy()

def get_fraud_catch_rate():
    """Business metric: % of fraud caught"""
    m = global_model_metrics
    if m["total_fraud"] == 0:
        return 0.0
    return round(m["fraud_caught"] / m["total_fraud"] * 100, 1)

def get_precision_rate():
    """Business metric: % of flagged transactions that were actually fraud"""
    m = global_model_metrics
    if m["total_blocked"] == 0:
        return 100.0
    true_positives = m["total_blocked"] - m["false_positives"]
    return round(true_positives / m["total_blocked"] * 100, 1)

print("\n✓ Engine ready with metrics export functions")
print(f"✓ Current AUC-ROC: {global_model_metrics['auc_roc']}")
print(f"✓ Current F1 Score: {global_model_metrics['f1']}")
print(f"✓ Fraud catch rate: {get_fraud_catch_rate()}%")
print(f"✓ Precision: {get_precision_rate()}%")