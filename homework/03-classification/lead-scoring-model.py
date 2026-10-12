import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def print_section(section):
    print('\n---------- %s -----------\n' %section)

# data - course lead scoring 2026, target - 'converted'
df = pd.read_csv('./data.csv')
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

print_section('industry observation')
print(df.industry.value_counts())
# Q1 - technology is most frequent

print_section('correlation matrix')
with pd.option_context('display.max_columns', None,
                       'display.max_rows', None,
                       'display.width', None,
                       'display.max_colwidth', None):
    print(df[numerical].corr().round(2))
# Q2 - lead_score & interaction_count have the biggest correlation

# splitting the data

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train['converted'].values
y_train = df_train['converted'].values
y_val = df_val['converted'].values
y_test = df_test['converted'].values

# del df_full_train['converted']
# del df_train['converted']
# del df_val['converted']
# del df_test['converted']

print_section('mutual information scores')

def get_mi(series):
    return mutual_info_score(series, df_train.converted)

mi_scores = df_train[categorical].apply(get_mi)
print(mi_scores.sort_values(ascending=False).round(2))
# Q3 - lead_source has the biggest MI score

# training the model
print_section('training the model')

dv = DictVectorizer(sparse=False)

dicts_train = df_train[categorical + numerical].to_dict(orient='records')
dicts_val = df_val[categorical + numerical].to_dict(orient='records')

X_train = dv.fit_transform(dicts_train)
X_val = dv.transform(dicts_val)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

accuracy = (y_pred == y_val).mean()
print("model accuracy: ", round(accuracy, 10))
# Q4 - 0.65 is the accuracy

# calculating accuracies for models trained on shrinked datasets
print_section('accuracy with eliminated features')

def get_model_accuracy(eliminated_feature='', C=0.1):
    remaining_features = (categorical + numerical)
    if eliminated_feature != '':
        remaining_features.remove(eliminated_feature)

    dv = DictVectorizer(sparse=False)

    dicts_train = df_train[remaining_features].to_dict(orient='records')
    dicts_val = df_val[remaining_features].to_dict(orient='records')

    X_train = dv.fit_transform(dicts_train)
    X_val = dv.transform(dicts_val)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    return (y_pred == y_val).mean()

for c in (categorical + numerical):
    current_accuracy = get_model_accuracy(eliminated_feature=c)
    print(abs(accuracy - current_accuracy).round(3), ' - accuracy difference for %s' %c)
# Q5 - eliminating number_of_courses_viewed gives the least difference

# trying regularization parameters
print_section('regularization parameters')

C_options = [0.000001, 0.00001, 0.0001, 0.001]
for C in C_options:
    current_accuracy = get_model_accuracy(C=C)
    print('C = ', C, '; accuracy = ', round(current_accuracy, 3))

# Q6 - C =  0.001 gives best accuracy

print_section("using the model")

dv = DictVectorizer(sparse=False)

dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')
dicts_test = df_test[categorical + numerical].to_dict(orient='records')

dv.fit(dicts_full_train)

X_full_train = dv.transform(dicts_full_train)
X_test = dv.transform(dicts_test)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_full_train, y_full_train)

y_test_pred = model.predict(X_test)
y_test_probability = model.predict_proba(X_test)[:, 1]
accuracy = (y_test == y_test_pred).mean()

print('accuracy: ', accuracy)

case_idx = 7
case = dicts_test[case_idx]

print('\ncase #%s:\n' %case_idx, case)

print()
print('real target value: ', y_test[case_idx])
print('target soft pred: ', y_test_probability[case_idx].round(2))
print('target hard pred: ', y_test_pred[case_idx])




