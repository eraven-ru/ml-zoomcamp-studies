import pandas as pd
import numpy as np
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

depths = [None, 1, 2, 4, 5, 6, 10, 15]
samples = [1, 2, 5, 15, 20, 100, 200]

for d in depths:
    for s in samples:
        score_decision_tree(max_depth=d, min_samples_leaf=s)

df_scores = pd.DataFrame(scores, columns=['max_depth', 'min_samples_leaf', 'auc'])
df_scores_table = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values='auc')

sns.heatmap(df_scores_table, annot=True, fmt='.3f')
plt.show()

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