import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor

def print_section(section):
    print('\n---------- %s -----------\n' %section)

def rmse(y, y_pred):
    e = y - y_pred
    se = e ** 2
    mse = se.mean()

    return np.sqrt(mse)


data = pd.read_csv('./data.csv')
df = pd.DataFrame(data)
target = 'fuel_efficiency_mpg'

df = df.fillna(0)

print_section('df head')
print(df.head())

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = df_train[target].values
y_val = df_val[target].values

del df_train[target]
del df_val[target]

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

dv = DictVectorizer(sparse=True)
dv.fit(train_dicts)

X_train = dv.transform(train_dicts)
X_val = dv.transform(val_dicts)

print_section('tree w/ depth=1')

dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)

print(export_text(dt, feature_names=dv.get_feature_names_out()))
# Q1 - model_year is used in 1-leveled dt

print_section('random forest')

rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
error = rmse(y_val, y_pred)

print('y_pred: ', y_pred)
print('rmse: ', error)
# Q2 - 1.833 is the rmse

print_section('n_estimators tuning')

def n_estimators_tuning():
    for n in [10, 50, 100, 150]:
        rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)
        error = rmse(y_val, y_pred)

        print('n_estimators=%3d; rmse=%.3f' % (n, error))

n_estimators_tuning()
# Q3 - 150 estimators gives best rmse

print_section('max_depth tuning')

def max_depth_tuning():
    scores = []

    for d in [10, 15, 20, 25]:
        for n in [10, 50, 100, 150]:
            rf = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=1, n_jobs=-1)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_val)
            error = rmse(y_val, y_pred)
            scores.append((d, n, error))

    df_scores = pd.DataFrame(scores, columns=['max_depth', 'n_estimators', 'rmse'])

    for d in [10, 15, 20, 25]:
        df_subset = df_scores[df_scores.max_depth == d]
        mean_rmse = df_subset.rmse.mean()

        print('max_depth=%d; rmse=%.3f' % (d, mean_rmse))

max_depth_tuning()
# Q4 - max_depth=10 gives best mean rmse

print_section('feature_importance')

rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

feature_importances_zipped = zip(dv.get_feature_names_out(), rf.feature_importances_)
feature_importances = list(feature_importances_zipped)

df_fi = pd.DataFrame(feature_importances, columns=['feature', 'importance_score'])
print(df_fi.sort_values('importance_score', ascending=False))
# Q5 - vehicle_weight is the most important

print_section('boosting')

features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

watchlist = [(dtrain, 'train'), (dval, 'val')]
xgb_params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist)
# eta = 0.3; rmse = 1.82646
# eta = 0.1; rmse = 1.72481 - Q6 - this one is better





