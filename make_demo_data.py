import pandas as pd

df = pd.read_csv("train_transaction.csv")
demo = df.sample(5000, random_state=42)
demo.to_csv("demo_transactions.csv", index=False)

print("demo_transactions.csv created successfully")
print(f"Rows: {len(demo)}")