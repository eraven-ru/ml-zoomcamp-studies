import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def print_section(section):
    print('\n---------- %s -----------\n' %section)

# copied homework/03-classification

# data - course lead scoring 2026, target - 'converted'
df = pd.read_csv('../03-classification/data.csv')
print(df.head().T)

# columns & values already normalized

col_summary = pd.concat([
    df.dtypes.rename('dtype'),
    df.nunique().rename('nunique'),
], axis=1)

print_section("column summary")
print(col_summary)
# --> splitting categorical & numerical; here we do not have categorical with numerical types - splitting by type is ok

categorical = ['lead_source', 'industry', 'employment_status', 'location']
numerical = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']

print_section("is null")
print(df.isnull().sum())

df[categorical] = df[categorical].fillna("NA")
df[numerical] = df[numerical].fillna(0)

# splitting the data

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = df_train.converted.values
y_val = df_val.converted.values

# ROC AUC feature importance
print_section("ROC AUC feature importance")

for c in numerical:
    score = roc_auc_score(df_train.converted, df_train[c])
    if score < 0.5:
        score = roc_auc_score(df_train.converted, -df_train[c])
    print('roc auc = ', score, 'for %s' %c)

# Q1 - lead_score has the biggest roc auc score

# training the model
print_section('training the model -> auc score')

dicts_train = df_train[categorical + numerical].to_dict(orient='records')
dicts_val = df_val[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(dicts_train)
X_val = dv.transform(dicts_val)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_pred)

print('model auc score: ', round(auc_score, 3))
# Q2 - 0.732 is the auc score

print_section('plotting precision and recall')

precision = []
recall = []
f1 = []

thresholds = np.arange(0, 1 + 0.01, 0.01)

for t in thresholds:
    actual_positive = y_val == 1
    actual_negative = y_val == 0

    predicted_positive = y_pred >= t
    predicted_negative = y_pred < t

    tn = (actual_negative & predicted_negative).sum()
    fp = (actual_negative & predicted_positive).sum()

    fn = (actual_positive & predicted_negative).sum()
    tp = (actual_positive & predicted_positive).sum()

    p = tp / (fp + tp) if (fp + tp) > 0 else 0
    r = tp / (fn + tp) if (fn + tp) > 0 else 0
    f1_score = (2 * p * r) / (p + r) if (p + r) > 0 else 0

    precision.append(p)
    recall.append(r)
    f1.append(f1_score)

plt.plot(thresholds, precision, label='p', color='blue')
plt.plot(thresholds, recall, label='r', color='green')
plt.plot(thresholds, f1, label='f1', color='violet')
plt.legend()
plt.grid(True)
plt.show()

# Q3 - intersection is at 0.63

f1_max_idx = np.argmax(f1)
f1_max_threshold = thresholds[f1_max_idx]
print('f1_max_threshold: ', f1_max_threshold)

# Q4 - 0.41 is t on which f1 is max

print_section('cross validation')

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.converted.values
    y_val = df_val.converted.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    score = roc_auc_score(y_val, y_pred)
    scores.append(score)

print('auc scores std: %.3f' % np.std(scores))
# Q5 - 0.007 is std

print_section('hyperparameter tuning')

for C in [0.000001, 0.001, 1]:
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.converted.values
        y_val = df_val.converted.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        score = roc_auc_score(y_val, y_pred)
        scores.append(score)

    print('C = %.6f; roc_auc mean = %.3f, roc_auc std = %.3f' % (C, np.mean(scores), np.std(scores)))

# Q6 - 0.001 gives best mean



