import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def print_section(section):
    print('\n---------- %s -----------\n' %section)

# Section:  1 - copy relevant model construction from 03

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

# prepare feature matrices

dv = DictVectorizer(sparse=False)
train_dict = df_train[categorical_cols + numerical_cols].to_dict(orient='records')
val_dict = df_val[categorical_cols + numerical_cols].to_dict(orient='records')

dv.fit(train_dict)

X_train = dv.transform(train_dict)
X_val = dv.transform(val_dict)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# training a model

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred_hard = model.predict(X_val)
y_pred = model.predict_proba(X_val)

# training on full train
def full_train():
    dicts_full_train = df_full_train[categorical_cols + numerical_cols].to_dict(orient='records')
    dicts_test = df_test[categorical_cols + numerical_cols].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    dv.fit(dicts_full_train)

    X_full_train = dv.transform(dicts_full_train)
    X_test = dv.transform(dicts_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_full_train, y_full_train)

    y_test_pred = model.predict_proba(X_test)[:, 1]
    churn_decision = y_test_pred >= 0.5

# Section: Analyzing Accuracy metric for binary classification problems
# accuracy - basically a share of correct predictions out of all

y_pred = y_pred[:, 1]
churn_decision = y_pred >= 0.5

accuracy = accuracy_score(y_val, churn_decision)
# same as (churn_decision == y_val).mean()

print_section("evaluating binary classification models")
print("model accuracy: ", accuracy)

# NB: here we picked 0.5 as a threshold, but it isn't always the best

def print_thresholds():
    scores = []
    thresholds = np.linspace(0, 1.0, 21)

    print_section("thresholds comparison")
    for t in thresholds:
        score = accuracy_score(y_val, y_pred >= t)
        print("%.2f %.3f" % (t, score))
        scores.append(score)

    plt.plot(thresholds, scores)
    plt.show()

# print_thresholds()

# ii: here we see 0.5 is best, but it's not for any model - better to check

# ii: dummy model - model giving just one result
churn_decision = y_pred >= 1.0
print('dummy accuracy: ', accuracy_score(y_val, churn_decision))

# --> seeing that dummy's accuracy is just 7-8% worse than our model's
# shows that accuracy metric might not be the best

# ii: such small difference is due to the class imbalance in the set - not churning is 3 times more likely
# dummy essentially predicts most probable result

# Section: Confusion Tables

# for binary classification we have 4 different types of final model decisions
# ii: 1 - predicted values are split into Positive (decision = 1) & Negative (decision = 0) Classes
# then based on actual cases we would have:

# True Positive (y_val = 1 & decision = 1)
# True Negative (y_val = 0 & decision = 0)
# False Positive (y_val = 0 & decision = 1)
# False Negative (y_val = 1 & decision = 0)

t = 0.5 # threshold

actual_positive = (y_val == 1)
actual_negative = (y_val == 0)

predicted_positive = (y_pred >= t)
predicted_negative = (y_pred < t)

tp = (actual_positive & predicted_positive).sum()
tn = (actual_negative & predicted_negative).sum()

fp = (actual_negative & predicted_positive).sum()
fn = (actual_positive & predicted_negative).sum()

confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])

confusion_matrix_p = (confusion_matrix / confusion_matrix.sum()).round(2) # calculate percentages

print_section("confusion matrix")
print(confusion_matrix, '\n')
print(confusion_matrix_p)
# --> get ~8% false positive & 12% false negative

# Section: Precision & Recall
# - 2 additional metrics for evaluating models derived from confusion tables

print_section("precision & recall")

# ii: Precision - fraction of correctly identified positives in Positive class
# --> in our case: fraction of users who actually churned out of those who we predicted would churn

precision = tp / (tp + fp)
print("precision: ", precision)
print("1 - precision: ", 1 - precision, '\n')
# ~33% of assumed churning did not churn - lost money by offering discounts to wrong users

# ii: Recall - fraction of identified (predicted) positives out of actual positives
# --> in our case: out of all who actually churned we get fraction of those who we predicted would churn

recall = tp / (tp + fn)
print("recall: ", recall)
print("1 - recall: ", 1 - recall)
# ~ 45% of those who churned were identified

# --> based on precision & recall the model feels much worse than from seeing accuracy alone (80%)

# Section: ROC curves / ROC - Receiver Operating Characteristic
# tpr & fpr for real, random & ideal models plot for thresholds (0..1), then against each other

print_section("ROC curves")
print('# uncomment to show the plots')

# ii: tpr = recall - true positive rate, out of real positives how much we predicted
tpr = tp / (tp + fn) # we want it as big as possible

# ii: fpr - false positive rate, out of real negatives how much we predicted as positives
fpr = fp / (tn + fp) # we want it as small as possible

thresholds = np.linspace(0, 1, 101)

def tpr_fpr_scores(y_val, y_pred):
    scores = []

    for t in thresholds:
        actual_positive = y_val == 1
        actual_negative = y_val == 0

        predicted_positive = y_pred >= t
        predicted_negative = y_pred < t

        tp = (actual_positive & predicted_positive).sum()
        tn = (actual_negative & predicted_negative).sum()

        fp = (actual_negative & predicted_positive).sum()
        fn = (actual_positive & predicted_negative).sum()

        scores.append((t, tp, tn, fp, fn))

    columns = ['threshold', 'tp', 'tn', 'fp', 'fn']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)

    return df_scores

def show_plot(df_scores):
    plt.plot(thresholds, df_scores.tpr, label='tpr')
    plt.plot(thresholds, df_scores.fpr, label='fpr')
    plt.legend()
    plt.show()

df_scores = tpr_fpr_scores(y_val, y_pred)
# show_plot(df_scores)

# to understand the trends we should compare with random & ideal model

np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))

df_rand = tpr_fpr_scores(y_val, y_rand)
# show_plot(df_rand)
# shows diagonal line

y_ideal = np.repeat([0, 1], [(y_val == 0).sum(), (y_val == 1).sum()])
y_ideal_pred = np.linspace(0, 1, len(y_ideal))

df_ideal = tpr_fpr_scores(y_ideal, y_ideal_pred)
# show_plot(df_ideal)
# shows ideal tpr & fpr at t = (1 - (y_val == 1).mean())

# ii: ROC curve is a plot of tpr by fpr

plt.figure(figsize=(5, 5))

# plt.plot(df_scores.fpr, df_scores.tpr, color='blue', label='model')
# --> there is a sklearn function
fpr, tpr, threshold = roc_curve(y_val, y_pred)
plt.plot(fpr, tpr, color='blue', label='model')

plt.plot(df_ideal.fpr, df_ideal.tpr, color='green', label='ideal')
plt.plot(df_rand.fpr, df_rand.tpr, color='red', label='rand')

plt.legend()
# plt.show()

# NB: good model is a model that is closes to ideal and further from random
# if the curve is below random - there is a mistake

# Section: AUC - Area Under the Curve
# - metric that is derived from ROC curve

print_section("auc score")

fpr, tpr, threshold = roc_curve(y_val, y_pred)
auc_score = auc(fpr, tpr)

# OR

auc_score = roc_auc_score(y_val, y_pred)

print("auc_score: ", auc_score)

# ii: AUC interpretation - a probability of randomly selected positive example's score
#  being higher than randomly selected negative example's one

n = 100000

pos = y_pred[y_val == 1]
neg = y_pred[y_val == 0]

pos_idx = np.random.randint(0, len(pos), n)
neg_idx = np.random.randint(0, len(neg), n)

auc_interpretation = (pos[pos_idx] > neg[neg_idx]).mean()
print("auc_interpretation: ", auc_interpretation)

# Section: Cross-Validation
# - method of splitting full-train dataset (train + val) differently
# - full-train is split into k folds & sent to k training rounds with each fold as a val-data in turn

# ii: cross-validation is used for parameter-tuning (ie C - regularization) by calculating mean & std of roc_auc
# ii: use cross-validation on smaller datasets or to see the stability of the model

# 1: train & predict methods

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

# KFold

print_section('parameter tuning (cross-validation)')
print('# uncomment to show the comparison')

def compare_parameters():
    for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
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
        # --> no big variations of auc - the model & data are pretty stable, C=1.0 is best

# compare_parameters()

# final training

model, dv = train(df_full_train, y_full_train, C=1.0)
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)

print_section('final training')
print('final auc score: ', auc)




