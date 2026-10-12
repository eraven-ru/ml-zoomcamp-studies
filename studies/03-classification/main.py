import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# Section: Classification Problem

# - determine which telecom company's customer is going to churn
# g(xi) =~ yi where xi - information abt i's customer, y - 0 or 1 (1 if churned)
# - binary classification

df = pd.read_csv("./data.csv")

# Section: Preparing Data
# - normalize columns and rows, alter types for totalcharges & churn

df.columns = df.columns.str.lower().str.replace(" ", "_")
categorical_cols = list(df.dtypes[(df.dtypes == "str") | (df.dtypes == "object")].index)

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

y_full_train = df_full_train["churn"].values
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

global_churn_rate = df_full_train.churn.mean()

def print_eda():
    print(df_full_train.isnull().sum(), "\n") # no null values left
    print(df_full_train.churn.value_counts(normalize=True), "\n")

    print("global_churn_rate: ", round(global_churn_rate, 2), "\n")

    print("numerical_cols: ", numerical_cols)
    print("categorical_cols: ", categorical_cols, "\n")
    print(df_full_train[categorical_cols].nunique())

print_eda()

# Section: EDA: Feature Importance

# metrics for feature importance are calculated relative to group_churn_rate & global_churn_rate
# 1st metric - difference = group_churn_rate - global_churn_rate (> 0 means group more likely to churn)
# 2nd metric - risk = group_churn_rate / global_churn_rate (> 1 means group more likely to churn)

female_churn_rate = df_full_train[df_full_train.gender == 'female'].churn.mean()
male_churn_rate = df_full_train[df_full_train.gender == 'male'].churn.mean()
# --> shows little difference (less than 1%)

partner_churn_rate = df_full_train[df_full_train.partner == 'yes'].churn.mean()
no_partner_churn_rate = df_full_train[df_full_train.partner == 'no'].churn.mean()
print('\n', (round(global_churn_rate, 2), round(partner_churn_rate, 2), round(no_partner_churn_rate, 2)))
# --> shows more significant difference (5-6%) - "no_partner" group is more likely to churn

def print_feature_importance():
    for c in categorical_cols:
        df_group = df_full_train.groupby(c).churn.agg(['count', 'mean'])
        df_group['diff'] = df_group['mean'] - global_churn_rate
        df_group['risk'] = df_group['mean'] / global_churn_rate

        print('\n', c)
        print(df_group)

print_feature_importance()

# Section: EDA: Feature Importance: Mutual Information

# ii: Mutual Information (MI) - measure of informational dependence of 2 random variables
# - how much (in bits or other info metrics) do we learn about 1 variable by knowing the value of another

# mi - is a way to measure feature importance for categorical variables

def get_mi(series):
    return mutual_info_score(series, df_full_train.churn)

def print_feature_mi():
    mi_scores = df_full_train[categorical_cols].apply(get_mi)
    print(mi_scores.sort_values(ascending=False))

print("\n-------- MI scores ---------\n")
print_feature_mi()

# NB: the bigger the feature MI, the more important the feature is for training

# Section: EDA: Feature Importance: Correlation

# Pearson correlation coefficient (r) measures linear dependency between 2 numeric variables
# -1 <= r <= 1

# sign of r tells us the direction of the relationship, the absolute value - its strength
# abs(r) - [0, 0.2] - low; [0.2, 0.5] - moderate; [0.5, 1] - strong;

# r - way to measure feature importance of numeric variables

def print_feature_correlation():
    print(df_full_train[numerical_cols].corrwith(df_full_train.churn))

print("\n-------- Correlation ---------\n")
print_feature_correlation()
# --> tenure has the strongest negative correlation (the bigger the tenure - the less likely to churn)

# Section: One-Hot Encoding
# - process of transforming categorical columns in feature matrix's binary columns (already done in 02)

# transform df with features to dictionary, fit the DictVectorizer, transform the feature columns

dv = DictVectorizer(sparse=False) # sparse matrices optimize zero-rows or cols
train_dict = df_train[categorical_cols + numerical_cols].to_dict(orient='records')
val_dict = df_val[categorical_cols + numerical_cols].to_dict(orient='records')

dv.fit(train_dict) # determines all cols, including binary

X_train = dv.transform(train_dict) # numerical are not transformed
X_val = dv.transform(val_dict)

# dv.feature_names_ to see all cols including binary

# Section: Logistic Regression

# g(xi) =~ yi, where yi {0, 1}
# in these cases we use logistic regression - g(xi) = sigmoid(w0 + W.T.dot(xi))
# ---> result of linear regression is taken as a "score" of which sigmoid is calculated

# sigmoid(z) = 1 / (1 + exp(-z)), always in range (0, 1), infinitely approaching 0 & 1 on -inf and +inf, = 0.5 at 0
# NB: Logistic regression (with linear r.) is a linear model - cause it uses liner regression formula

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(xi, w):
    score = w[0]

    for j in range(len(xi)):
        score += w[j] * xi[j]

    result = sigmoid(score)
    return result

# Section: Training / Fitting Logistic Regression

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

print('\n-------- logistic regression model ---------\n')
print("bias term: ", model.intercept_[0].round(3))
print("coefficients: ", model.coef_[0].round(3), '\n')

y_pred_hard = model.predict(X_val) # already transformed into 0's & 1's through threshold = 0.5
y_pred = model.predict_proba(X_val) # soft prediction, probability between 0 & 1

# y_pred returns [[pred1, pred2], [...], ...] - pred1 - probability of 0, pred2 - probability of 1
# --> take only pred2 - y_pred[:, 1]

churn_decision = y_pred[:, 1] > 0.5

# we measure the performance with accuracy - share of getting it right
print("accuracy: ", (churn_decision == y_val).mean())

# Section: Model Interpretation
# - interpreting bias and coefficients

# train a smaller model; create test customers; learn and interpret their scores relative to bias & coef;

columns_for_small = ['contract', 'tenure', 'monthlycharges']
df_train_small = df_train[columns_for_small]
df_val_small = df_val[columns_for_small]

dict_train_small = df_train_small.to_dict(orient='records')
dict_val_small = df_val_small.to_dict(orient='records')

dv_small = DictVectorizer(sparse=False)
dv_small.fit(dict_train_small)

X_train_small = dv_small.transform(dict_train_small)
X_val_small = dv_small.transform(dict_val_small)

model_small = LogisticRegression(max_iter=1000)
model_small.fit(X_train_small, y_train)

print("\n--------- model interpretation (on smaller df) ---------\n")

print("bias term: ", model_small.intercept_[0].round(3))
print("coef: ")
pprint.pprint(dict(zip(dv_small.get_feature_names_out(), model_small.coef_[0].round(3))))
print()

customer_1 = { 'contract': 'month_to_month', 'tenure': 2, 'monthlycharges': 60 }
customer_2 = { 'contract': 'one_year', 'tenure': 12, 'monthlycharges': 40 }

customer_1_pred = sigmoid(-2.478 + 0.971 + 0.027 * 60 + (-0.036) * 2)
customer_2_pred = sigmoid(-2.478 + (-0.024) + 0.027 * 40 + (-0.036) * 12)
# NB: if score (part inside sigmoid) is > 0 => we get probability of churning > 0.5

print("bias prediction: ", sigmoid(model_small.intercept_[0]), '\n')

print("customer_1: ", customer_1)
print("customer_1 prediction: ", customer_1_pred, '\n')

print("customer_2: ", customer_2)
print("customer_2 prediction: ", customer_2_pred, '\n')

# accuracy of a smaller model
y_small_pred = model_small.predict_proba(X_val_small)[:, 1] > 0.5
print("model_small accuracy: ", (y_small_pred == y_val).mean())

# Section: Using the Model

# train the model on full train; test on df_test; use on several customers from test dataset;

print("\n--------- model use ---------\n")

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

print("full model accuracy: ", (churn_decision == y_test).mean().round(3), '\n') # slightly better

customer_idx = 59
customer = dicts_test[customer_idx]
X_customer = dv.transform([customer])

churn_probability = model.predict_proba(X_customer)[0, 1]

print('IDX: %s' %customer_idx)
pprint.pprint(customer)

print('\nreal churn value: ', y_test[customer_idx])
print('pred churn value: ', churn_probability)




