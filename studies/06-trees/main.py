import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def print_section(section):
    print('\n------------- %s -------------\n' %section)

# Section: The problem

# - based on bank's information about a customer & his debt
# determine the risk of him not paying it back ("defaulting")

df = pd.read_csv('./CreditScoring.csv')

# Section: Data Preparation

# here:
# columns -> lowercase; turn categorical values into meaningful strings;
# change 9999..99 from numerical cols -> NaN; split the dataset & prepare X & y for training;

df.columns = df.columns.str.lower()

categorical = ['home', 'marital', 'records', 'job']
numerical = ['seniority', 'time', 'age', 'expenses', 'income', 'assets', 'debt', 'amount', 'price']
target = 'status'

df['status'] = df.status.map({
    1: 'ok',
    2: 'default',
    0: 'unknown'
})

df['home'] = df['home'].map({
    1: "rent",
    2: "owner",
    3: "priv",
    4: "ignore",
    5: "parents",
    6: "other"
})

df['marital'] = df['marital'].map({
    1: "single",
    2: "married",
    3: "widow",
    4: "separated",
    5: "divorced"
})

df['records'] = df['records'].map({
    1: 'no_rec',
    2: 'yes_rec'
})

df['job'] = df['job'].map({
    1: "fixed",
    2: "partime",
    3: "freelance",
    4: "others"
})

for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)

df = df[df.status != 'unknown'] # not interested in rows with unknown target
df = df.reset_index(drop=True)

print_section('df head')
print(df.head())

# splitting

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = (df_train[target] == 'default').astype(int).values
y_val = (df_val[target] == 'default').astype(int).values
y_test = (df_test[target] == 'default').astype(int).values

del df_train[target]
del df_val[target]
del df_test[target]

# Section: Decision Trees

# todo: build a primitive decision tree; then use sklearn to train a basic tree model;
# todo: then compare different max_depth's & their auc scores & print their conditions;

# ii: Decision Tree algorithm - an algorithm based on a tree structure
# ii: which has conditions as nodes & decisions as leaves

print_section('primitive decision tree')

def primitive_decision_tree(client):
    if client['records'] == 'yes':
        if client['job'] == 'parttime':
            return 'default'
        else:
            return 'ok'
    else:
        if client['assets'] > 6000:
            return 'ok'
        else:
            return 'default'

client_idx = 10
client = df_train.iloc[client_idx].to_dict()
print('client #%d: ' %client_idx, client, '\n')
print('decision: ', primitive_decision_tree(client))

print_section('decision trees')

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dicts)

X_train = dv.transform(train_dicts)
X_val = dv.transform(val_dicts)

def train_decision_tree(max_depth=None):
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    y_pred = dt.predict_proba(X_val)[:, 1]
    y_train_pred = dt.predict_proba(X_train)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_pred)

    print("--> tree w/ max_depth = %s" %max_depth, '\n')
    print('train auc: ', train_auc)
    print('val auc: ', val_auc, '\n')

    if max_depth is not None:
        if (max_depth > 0) & (max_depth < 4):
            print(export_text(dt, feature_names=dv.get_feature_names_out()))

train_decision_tree()
# --> we see train_auc == 1 & val_auc =~ 0.65
# ii: Overfitting - state of a model when it memorized the training data
# ii: but fails to generalize; Decision trees are prone to overfitting
# here that happens because no max_depth is specified
# - a tree can basically grow deep enough to just encode specific cases ("memorize")

train_decision_tree(max_depth=1)

# ii: tree with a max_depth of 1 is called a "decision stump"
# - it just encodes 1 condition

for d in [2, 3, 4, 5]:
    train_decision_tree(max_depth=d)

# performance grows with bigger depths

# Section: Decision Trees Algorithm

# 2 parts of an algorithm are to be described:
# - defining the condition for each consecutive Condition Node
# - stopping criteria

# 1. Defining Conditions:

# - for each node we are searching for the best performing [feature > T] (T - threshold)
# - we define list of [features] & corresponding possible [T]'s - Ts are mostly values of features
# - for each [feature, T] we make a split in the dataset df[feature > T] & df[feature <= T]
# - for each such split we compute "Impurity"
#   Impurity is based on several metrics - Misclassification Rate, Entropy & Gini
#   weighted average of split parts' metrics is taken as Impurity
# - [feature, T] with lowest Impurity is taken as a Node Condition

# 2. Stopping Criteria

# we go with above algorithm building nodes top->down until stopping criteria for a split is met:

# - the node is "pure" - Impurity == 0 for current split
# - max_depth is reached
# - the split is too small to split further (limit is a controlled parameter)

# Section: Parameter Tuning

# todo: for max_depth & min_sample_leaf do parameter tuning
# - try various values & show heatmap for resulting auc scores

scores = []

def score_decision_tree(max_depth=None, min_samples_leaf=1):
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    dt.fit(X_train, y_train)

    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    scores.append((max_depth, min_samples_leaf, auc))

def dt_parameters_heatmap():
    depths = [None, 1, 2, 4, 5, 6, 10, 15]
    samples = [1, 2, 5, 15, 20, 100, 200]

    for d in depths:
        for s in samples:
            score_decision_tree(max_depth=d, min_samples_leaf=s)

    df_scores = pd.DataFrame(scores, columns=['max_depth', 'min_samples_leaf', 'auc'])
    df_scores_table = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values='auc')

    sns.heatmap(df_scores_table, annot=True, fmt='.3f')
    plt.show()

# dt_parameters_heatmap()
# min_samples_leaf = 15 is best
# max_depth has best scores at [None, 6, 10, 15] - we take 6 for simplicity

# here it is ok to try as many combinations cause the model is small, for larger datasets try less

dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)

# NB: why does decision tree return probability?

# Section: Ensemble Learning & Random Forest

# ii: idea - instead of 1 model train n models on different random subsets of features
# ii: & get their average outputs

# ii: Random Forest - ensemble of decision trees

# here: train a random forest + tune n_estimators (trees), max_depth & min_samples_leaf

def random_forest_tuning():
    def n_estimators_tuning():
        n_estimators_options = np.arange(10, 201, 10)
        n_estimators_scores = []

        for n in n_estimators_options:
            rf = RandomForestClassifier(n_estimators=n, random_state=1)
            rf.fit(X_train, y_train)

            y_pred = rf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            n_estimators_scores.append((n, auc))

        df_n_estimators_scores = pd.DataFrame(n_estimators_scores, columns=['n_estimators', 'auc'])
        plt.plot(df_n_estimators_scores.n_estimators, df_n_estimators_scores.auc)
        plt.show()

        print('n_estimators: good growth until 50, then only small improvement')

    def max_depth_tuning():
        max_depth_options = [5, 10, 15]
        n_estimators_options = np.arange(10, 201, 10)
        max_depth_scores = []

        for d in max_depth_options:
            for n in n_estimators_options:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
                rf.fit(X_train, y_train)

                y_pred = rf.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred)
                max_depth_scores.append((d, n, auc))

        df_max_depth_scores = pd.DataFrame(max_depth_scores, columns=['max_depth', 'n_estimators', 'auc'])

        for d in max_depth_options:
            df_subset = df_max_depth_scores[df_max_depth_scores.max_depth == d]
            plt.plot(df_subset.n_estimators, df_subset.auc, label='max_depth=%d' % d)

        plt.legend()
        plt.show()

        print('max_depth: value of 10 gives best results')

    max_depth = 10

    def min_samples_tuning():
        min_samples_options = [1, 3, 5, 10, 15]
        n_estimators_options = np.arange(10, 201, 10)
        min_samples_scores = []

        for s in min_samples_options:
            for n in n_estimators_options:
                rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth, min_samples_leaf=s, random_state=1)
                rf.fit(X_train, y_train)

                y_pred = rf.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred)
                min_samples_scores.append((s, n, auc))

        df_min_samples_scores = pd.DataFrame(min_samples_scores, columns=['min_samples', 'n_estimators', 'auc'])

        for s in min_samples_options:
            df_subset = df_min_samples_scores[df_min_samples_scores.min_samples == s]
            plt.plot(df_subset.n_estimators, df_subset.auc, label='min_samples=%d' % s)

        plt.legend()
        plt.show()

        print('min_samples_leaf: 1 & 3 give best results')

    n_estimators_tuning()
    max_depth_tuning()
    min_samples_tuning()

    print('(1 & 2 graphs look different from the example, look into it)')

print_section('random forest tuning')
# random_forest_tuning()

n_estimators = 100
max_depth = 10
min_samples_leaf = 3

rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    random_state=1
)

rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)

print()
print('best rf score: ', auc)

# Section: Gradient Boosting

# ii: boosting is an algorithm for ensemble learning where models are trained sequentially
# ii: each learning on mistakes of the previous model

# ii: Gradient Boosting Trees is a boosting algorithm where all models are trees

# todo: train an xgboost model, print its progress by rounds, parse the results & plot train_auc & val_auc

print_section('gradient boosting trees')

features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

watchlist = [(dtrain, 'train'), (dval, 'val')]
watchlist_result = {}
xgb_params = {
    'eta': 0.3, # learning rate
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8, # n of cores

    'seed': 1,
}

model = xgb.train(xgb_params, dtrain,
                  num_boost_round=100,
                  evals=watchlist,
                  evals_result=watchlist_result,
                  verbose_eval=5
)

y_pred = model.predict(dval)
auc = roc_auc_score(y_val, y_pred)

print()
print('auc (xgboost model w/ 100 rounds): ', auc)

def plot_xgb_iteration_aucs():
    train_aucs = list(watchlist_result['train'].values())[0]
    val_aucs = list(watchlist_result['val'].values())[0]

    scores_data = list(zip(range(1, len(train_aucs) + 1), train_aucs, val_aucs))
    df_scores = pd.DataFrame(scores_data, columns=['iteration', 'train_auc', 'val_auc'])

    plt.plot(df_scores.iteration, df_scores.train_auc, label='train')
    plt.plot(df_scores.iteration, df_scores.val_auc, label='val')
    plt.legend()
    plt.show()

# plot_xgb_iteration_aucs()
# val_auc stops improving after ~10-25, declines after 35
# train auc is perfect after ~40 - clear sign of overfitting
# NB: Gradient Boosting models are prone to overfitting - it is important to limit parameters

# Section: XGBoost Parameter Tuning

# todo: most important parameters are 'eta', 'max_depth' & 'min_child_weight'
# - they are tuned in that order

print_section('xgboost tuning')

def eta_tuning():
    etas = [0.01, 0.05, 0.1, 0.3, 1.0]
    scores = {}

    for eta in etas:
        eval_results = {}
        xgb_params = {
            'eta': eta,
            'max_depth': 6,
            'min_child_weight': 1,

            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'nthread': 8,  # n of cores

            'seed': 1,
        }

        model = xgb.train(xgb_params, dtrain,
                          num_boost_round=200,
                          evals=watchlist,
                          evals_result=eval_results,
                          verbose_eval=5
        )

        train_aucs = list(eval_results['train'].values())[0]
        val_aucs = list(eval_results['val'].values())[0]

        evals_data = list(zip(range(1, len(train_aucs) + 1), train_aucs, val_aucs))
        evals_df = pd.DataFrame(evals_data, columns=['iteration', 'train_auc', 'val_auc'])

        scores[f'eta={eta}'] = evals_df

    for eta in etas:
        evals_df = scores[f'eta={eta}']
        plt.plot(evals_df.iteration, evals_df.val_auc, label=f'eta={eta}')

    plt.legend()
    plt.show()

# eta_tuning()
# NB: we are interested in a value that gives faster rise (less rounds)
# NB: & and (!!) stays stable on subsequent rounds
# eta = 0.1 gives best performance based on that - reaches top on ~60, is stable till 200

eta = 0.1

def max_depth_tuning():
    max_depth_options = [3, 5, 10, 15]
    scores = {}

    for d in max_depth_options:
        eval_results = {}
        xgb_params = {
            'eta': eta,
            'max_depth': d,
            'min_child_weight': 1,

            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'nthread': 8,  # n of cores

            'seed': 1,
        }

        model = xgb.train(xgb_params, dtrain,
                          num_boost_round=200,
                          evals=watchlist,
                          evals_result=eval_results,
                          verbose_eval=5
        )

        train_aucs = list(eval_results['train'].values())[0]
        val_aucs = list(eval_results['val'].values())[0]

        evals_data = list(zip(range(1, len(train_aucs) + 1), train_aucs, val_aucs))
        evals_df = pd.DataFrame(evals_data, columns=['iteration', 'train_auc', 'val_auc'])

        scores[f'max_depth={d}'] = evals_df

    for d in max_depth_options:
        evals_df = scores[f'max_depth={d}']
        plt.plot(evals_df.iteration, evals_df.val_auc, label=f'max_depth={d}')

    plt.legend()
    plt.show()

# max_depth_tuning()
# max_depth = 3 takes a little longer to reach max, but its max is bigger & is stable till end

max_depth = 3

def min_child_weight_tuning():
    weight_options = [1, 5, 10, 30]
    scores = {}

    for w in weight_options:
        eval_results = {}
        xgb_params = {
            'eta': eta,
            'max_depth': max_depth,
            'min_child_weight': w,

            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'nthread': 8,  # n of cores

            'seed': 1,
        }

        model = xgb.train(xgb_params, dtrain,
                          num_boost_round=200,
                          evals=watchlist,
                          evals_result=eval_results,
                          verbose_eval=5
        )

        train_aucs = list(eval_results['train'].values())[0]
        val_aucs = list(eval_results['val'].values())[0]

        evals_data = list(zip(range(1, len(train_aucs) + 1), train_aucs, val_aucs))
        evals_df = pd.DataFrame(evals_data, columns=['iteration', 'train_auc', 'val_auc'])

        scores[f'min_child_weight={w}'] = evals_df

    for w in weight_options:
        evals_df = scores[f'min_child_weight={w}']
        plt.plot(evals_df.iteration, evals_df.val_auc, label=f'min_child_weight={w}')

    plt.legend()
    plt.ylim(0.81, 0.84)
    plt.show()

# min_child_weight_tuning()
# the difference is very small - default of 1 gives best value at ~175 rounds

min_child_weight = 1

xgb_params = {
    'eta': eta,
    'max_depth': max_depth,
    'min_child_weight': min_child_weight,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,  # n of cores

    'seed': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=175)
y_pred = model.predict(dval)

auc = roc_auc_score(y_val, y_pred)
print('auc (tuned: eta=0.1; max_depth=3; min_child_weigh=1; rounds=175): ', auc)

# Section: Selecting the best model

# ii: in this session we trained & tuned 3 models:
# - decision tree
# - random forest
# - gradient boosting trees

# todo: compare auc scores, then train the best one on the full-train dataset

print_section('selecting the best model')

# decision tree
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val, y_pred)
print('decision tree auc: ', auc)

# random forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=3,
    random_state=1
)

rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print('random forest auc: ', auc)

# xgboost
xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,  # n of cores

    'seed': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=175)
y_pred = model.predict(dval)

auc = roc_auc_score(y_val, y_pred)
print('gradient boosting auc: ', auc)

# NB: xgboost is a very good model & it often performs better than others
# but: it is harder to tune - has more significant parameters

y_full_train = (df_full_train[target] == 'default').astype(int).values
df_full_train = df_full_train.reset_index(drop=True)
del df_full_train[target]

full_train_dicts = df_full_train.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

dv = DictVectorizer(sparse=True)
dv.fit(full_train_dicts)

X_full_train = dv.transform(full_train_dicts)
X_test = dv.transform(test_dicts)

features = list(dv.get_feature_names_out())
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, feature_names=features)

xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'nthread': 8,

    'seed': 8
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=175)
y_pred = model.predict(dtest)

auc = roc_auc_score(y_test, y_pred)

print()
print('final xgboost model auc (test score): ', auc)
# final auc is close to val_auc - that means model is not overfitted & good at generalizing