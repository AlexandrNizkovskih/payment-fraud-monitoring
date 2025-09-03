# Payment Fraud Monitoring Service

##  Цель проекта
Снижение потерь от мошенничества: ранжирование транзакций по риску для ручной проверки и автоматических правил.

---

## 🛠️ Решение
- Все признаки числовые → **StandardScaler**
- Классификатор: **Logistic Regression** (class_weight='balanced')
- Тонкая настройка порога под Precision/Recall и Precision@k
- Интерфейсы: **API** и **CLI**; поддержан сценарий **Top-K** в выдаче

---

## 📊 Метрики (порог 0.5)
| Метрика   | Значение |
|-----------|----------|
| ROC-AUC   | 0.972 |
| PR-AUC    | 0.720 |
| Recall    | 0.918 |
| Precision | 0.061 |
| F1        | 0.114 |
| Accuracy  | 0.976 |

> Рекомендуется подбирать порог под Precision@k (например, топ-0.51% транзакций для ручной модерации).

---

##  Использование
~~~bash
python src/train.py --input data/creditcard.csv --output model/model.pkl
python predict.py --input new_transactions.csv --config config.yaml --output predictions.csv
~~~

## 🧩 Конфигурация (config.yaml)
~~~yaml
title: "Payment Fraud Monitoring"
model_path: "model/model.pkl"
target_col: "Class"
drop_cols: []
threshold: 0.9
top_k: 500
~~~

## 📂 Структура
payment_fraud_monitoring/
├── data/
├── model/
├── src/
├── config.yaml
├── report/
└── README.md

## ✅ Результат для заказчика
- Снижение ручной нагрузки за счёт ранжирования риска
- Управляемый компромисс между Loss и Review Cost через порог/Top-K