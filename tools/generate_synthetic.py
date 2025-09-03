import pandas as pd, numpy as np, os
np.random.seed(42)
n = 500
df = pd.DataFrame({
  "amount": np.round(np.random.exponential(60, n), 2),
  "time_delta": np.round(np.random.exponential(30, n), 2),
  "is_night": np.random.choice([0,1], n),
  "country_mismatch": np.random.choice([0,1], n, p=[0.95, 0.05])
})
os.makedirs("data", exist_ok=True)
df.to_csv("data/sample_input.csv", index=False)
print("saved: data/sample_input.csv")