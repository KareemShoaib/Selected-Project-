import pandas as pd

ROWS_TO_EXTRACT = 5000

df = pd.read_csv('csv/clean/train.csv')
df = df.head(ROWS_TO_EXTRACT)
df.to_csv('csv/small/train.csv', index=False)
print(f"Extracted {ROWS_TO_EXTRACT} rows from train.csv")