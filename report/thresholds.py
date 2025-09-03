import os, joblib, pandas as pd, numpy as np, yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

CFG = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
MODEL_PATH = CFG.get("model_path","model/model.pkl")
TARGET_COL = CFG.get("target_col","Class")

# авто-выбор train файла
cand_paths = ["data/train.csv", "data/creditcard.csv", "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"]
DATA_PATH = None
for p in cand_paths:
    if os.path.exists(p):
        DATA_PATH = p
        break
if DATA_PATH is None:
    print("Не найден train.csv для thresholds.py"); raise SystemExit(0)

df = pd.read_csv(DATA_PATH)
if TARGET_COL not in df.columns:
    print(f"TARGET_COL='{TARGET_COL}' не найден  thresholds.py пропущен."); raise SystemExit(0)

y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])
pipe = joblib.load(MODEL_PATH)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
proba = pipe.predict_proba(X_te)[:,1]

ths = np.linspace(0.05, 0.95, 19)
rows = []
for t in ths:
    pred = (proba >= t).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred, average='binary', zero_division=0)
    rows.append({"threshold": float(np.round(t,2)), "precision": float(prec), "recall": float(rec), "f1": float(f1)})
pd.DataFrame(rows).to_csv("report/thresholds.csv", index=False)

k = max(1, int(0.01 * len(proba)))  # top-1%
idx = np.argsort(proba)[::-1][:k]
prec_at_k = float((y_te.values[idx] == 1).mean())
with open("report/precision_at_k.txt","w",encoding="utf-8") as f:
    f.write(f"Precision@1% = {prec_at_k:.4f}\n")

print("Готово: report/thresholds.csv и report/precision_at_k.txt")