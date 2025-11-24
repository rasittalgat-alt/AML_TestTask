import pandas as pd

df = pd.read_csv("esf_sample_200k.csv")
print("Форма:", df.shape)
print("Колонки:", list(df.columns))
print(df.head(20))
