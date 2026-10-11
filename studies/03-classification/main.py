import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Section: Classification Problem

# - determine which telecom company's customer is going to churn
# g(xi) =~ yi where xi - information abt i's customer, y - 0 or 1 (1 if churned)
# - binary classification

df = pd.read_csv("./data.csv")

# Section: Preparing Data
# - normalize columns and rows, alter types for totalcharges & churn

df.columns = df.columns.str.lower().str.replace(" ", "_")
categorical_cols = list(df.dtypes[df.dtypes == "object"].index)

for c in categorical_cols:
    df[c] = df[c].str.lower().str.replace(" ", "_")

df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce").fillna(0)
df.churn = (df.churn == "yes").astype("int")

print(df.head().T)

# Section: Validation Framework

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train["churn"].values
y_val = df_val["churn"].values
y_test = df_test["churn"].values

del df_train["churn"]
del df_val["churn"]
del df_test["churn"]

print((len(df_train), len(df_val), len(df_test)))

# Section: EDA
# find how many null values; determine churn rate;
# split numerical & categorical columns; see n unique for categorical;

numerical_cols = ["tenure", "monthlycharges", "totalcharges"]
categorical_cols = [c for c in df.columns.values if c not in numerical_cols + ["customerid", "churn"]]

def eda():
    print(df_full_train.isnull().sum(), "\n") # no null values left
    print(df_full_train.churn.value_counts(normalize=True), "\n")

    churn_rate = df_full_train.churn.mean()
    print("churn_rate: ", round(churn_rate, 2), "\n")

    print("numerical_cols: ", numerical_cols)
    print("categorical_cols: ", categorical_cols, "\n")
    print(df_full_train[categorical_cols].nunique())

eda()