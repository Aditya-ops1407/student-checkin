"""
prepare_data.py
---------------
Cleans data/raw/Student_Depression_Dataset.csv -> data/student_mental_health.csv

Dataset: Student Depression Dataset (4,464 university students, Bangladesh).
CITE THE EXACT SOURCE YOU DOWNLOADED IT FROM in your report.

Design decisions (explain these in the report):
  * TARGET 'Risk' = 1 if Depression == 'Yes'.
  * 'Health_condition' is kept in a SEPARATE column set (see config.LEAKY) but is
    excluded from the main model: it separates the classes almost perfectly
    ('Critical' -> 100% Yes, 'Good' -> 100% No), i.e. it behaves like a proxy of
    the label (data leakage). train.py reports results with and without it.
  * 'Home_division' is dropped: Bangladesh-specific, not usable for other populations.
  * Exact duplicate rows are removed.
"""
import pandas as pd

RAW = "data/raw/Student_Depression_Dataset.csv"
OUT = "data/student_mental_health.csv"

df = pd.read_csv(RAW)
print("Raw shape:", df.shape)

for c in df.columns:
    if df[c].dtype != "int64" and df[c].dtype != "float64":
        df[c] = df[c].str.strip()

df["Risk"] = (df["Depression"] == "Yes").astype(int)
df = df.drop(columns=["Depression", "Home_division"])

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows")
print("Missing values:", int(df.isna().sum().sum()))
print("Risk balance:", df["Risk"].value_counts().to_dict())

df.to_csv(OUT, index=False)
print("Saved", OUT, df.shape)
