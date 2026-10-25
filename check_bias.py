import pandas as pd
d = pd.read_csv("data.csv")
d["has_scheme"] = d.url.str.startswith("http")
d["root_only"] = ~d.url.str.replace(r"^https?://", "", regex=True).str.contains("/")
print(d.groupby("label")[["has_scheme", "root_only"]].mean())