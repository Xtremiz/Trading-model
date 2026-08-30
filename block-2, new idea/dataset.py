import pandas as pd

df1 = pd.read_csv(r"F:\Git-Hub\Trading model\block-2, new idea\goldbuydata.csv")
df2 = pd.read_csv(r"F:\Git-Hub\Trading model\block-2, new idea\goldselldata.csv")
print(df1["signal"].value_counts())
print("="*50)
print(df2["signal"].value_counts())

