import pandas as pd

FILE = "malicious_phish.csv"
URL_COL = "url"
LABEL_COL = "type"
PHISHING = {"phishing"}
LEGIT = {"benign"}

df = pd.read_csv(FILE)
print(df[LABEL_COL].value_counts())

df = df[[URL_COL, LABEL_COL]].dropna()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().str.strip()
df = df[df[LABEL_COL].isin(PHISHING | LEGIT)]
df["label"] = df[LABEL_COL].isin(PHISHING).astype(int)
df = df.rename(columns={URL_COL: "url"})[["url", "label"]].drop_duplicates()
df.to_csv("data.csv", index=False)

print(df["label"].value_counts(), len(df))
for lab in (0, 1):
    print("\nlabel", lab, "examples:")
    print(df[df["label"] == lab]["url"].head(3).to_string(index=False))