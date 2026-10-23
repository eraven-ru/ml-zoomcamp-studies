import pickle

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# parameters

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'

def print_section(section):
    print('\n---------- %s -----------\n' %section)

df = pd.read_csv("../03-classification/data.csv")

df.columns = df.columns.str.lower().str.replace(" ", "_")
categorical_cols = list(df.dtypes[(df.dtypes == "str") | (df.dtypes == "object")].index)

for c in categorical_cols:
    df[c] = df[c].str.lower().str.replace(" ", "_")

df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce").fillna(0)
df.churn = (df.churn == "yes").astype("int")

# preparing datasets

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train["churn"].values
y_train = df_train["churn"].values
y_val = df_val["churn"].values
y_test = df_test["churn"].values

del df_train["churn"]
del df_val["churn"]
del df_test["churn"]

numerical_cols = ["tenure", "monthlycharges", "totalcharges"]
categorical_cols = [c for c in df.columns.values if c not in numerical_cols + ["customerid", "churn"]]

global_churn_rate = df_full_train.churn.mean()

# train & predict

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical_cols + numerical_cols].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X_train, y_train)

    return model, dv

def predict(df, dv, model):
    dicts = df[categorical_cols + numerical_cols].to_dict(orient='records')
    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

print_section('model validation')

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    model, dv = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C = %.3f; AUC = %.6f +- %.6f' % (C, np.mean(scores), np.std(scores)))


# final training

model, dv = train(df_full_train, y_full_train, C=1.0)
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)

print_section('training')
print('auc score: ', auc)

# exporting

with open(output_file, 'wb') as f_out:
    pickle.dump((model, dv), f_out)

print()






