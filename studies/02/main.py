import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./data.csv")

def normalize_string_series(s: pd.Series):
    return s.str.lower().str.replace(" ", "_")

df.columns = df.columns.str.lower().str.replace(" ", "_")

str_cols = list(df.dtypes[df.dtypes == "str"].index) # dtype of strings from csv is sometimes object & df.dtypes is a Series from which we can access index
for col in str_cols:
    df[col] = df[col].str.lower().str.replace(" ", "_")

print(df.head())
print()

# for different series find number and examples of unique values;

def print_unique_stats():
    print(df.isnull().sum())
    print(df.nunique())

    for col in df.columns:
        print(col)
        print(df[col].unique()[:10])
        print()

# print_unique_stats()

# SECTION: Exploratory Data Analysis; build a histogram for msrp;

def show_msrp_histplot():
    # sns.histplot(df.msrp, bins=50)
    # -> this would show "long tail" distribution where long tail is made of a share of extremely expensive cars
    # NB: ! long tails in row values tend to confuse ML models

    sns.histplot(np.log1p(df.msrp), bins=50)
    # log smoothes out the distribution (removes long tail) and turns it closer to normal distribution - models perform better on normal dist values in training
    # log1p() - is log() that does (+ 1) to every value to avoid -infinity on potential zeroes
    plt.show()

# show_msrp_histplot()

# SECTION: Validation Framework;

n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

print(n_train, n_val, n_test)

# NB: shuffle DF before splitting to avoid any patterns in original set
idx = np.arange(n)

np.random.seed(2)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]
df_full_train = pd.concat([df_train, df_val])

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)

y_train = np.log1p(df_train["msrp"].values)
y_val = np.log1p(df_val["msrp"].values)
y_test = np.log1p(df_test["msrp"].values)
y_full_train = np.concat([y_train, y_val])

# NB: never forget to remove the target from feature matrix X
del df_train["msrp"]
del df_val["msrp"]
del df_test["msrp"]
del df_full_train["msrp"]

print(df_train.head())
print(df_val.head())
print(df_test.head())

print()

print(y_train)
print(y_val)
print(y_test)

print(len(df_train), len(df_val), len(df_test))

# SECTION: Linear Regression (simple)

# constructing for 1 row & shortened feature vector - xi
# NB: linear regression formula - g(xi) = w0 + w1 * xi1 + w2 * xi2 + ... + wn * xin
# NB: w0 - bias term - the prediction we make without info abt a car

row = df_train.iloc[10]
xi = [row["engine_hp"], row["city_mpg"], row["popularity"]] # [453, 11, 86]

w0 = 7.17
w = [0.01, 0.04, 0.002]

def linear_regression_1(xi):
    prediction = w0

    for j in range(len(xi)):
        prediction += w[j] * xi[j]

    return prediction

prediction = np.expm1(linear_regression_1(xi))
print("prediction: ", prediction)

# NB: linear regression can be written as dot product Xw;
# we add [1, ..., 1] column to X for bias term (w0) multiplication

w = [w0] + w
X_1 = np.array([
    [1, 138, 24, 1385],
    [1, 124, 40, 700],
    [1, 453, 11, 86]
])

def linear_regression(X):
    return X.dot(w) # multiplying each row by w

print("predictions (dot version): ", linear_regression(X_1))

# Section: Training LR model

# training for g(X) ~ Xw formula is a process of finding (w) vector
# since we are targeting y and g(X) ~ Xw =~ y we would come to w =~ X_inverse * y
# NB: X does not have inverse as it is not a square matrix
# Gram matrix - X_T * X creates a square matrix from X
# (X_T * X) * w =~ X_T * y
# NB: w =~ (X_T * X)**(-1) * X_T * y

X = np.array([
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 86],
    [38,  54, 185],
    [142, 25, 431],
    [453, 31, 86],
])

y = [100, 200, 150, 250, 100, 200, 150, 250, 120]

def train_linear_regression(X, y):
    X = np.column_stack([np.ones(X.shape[0]), X])
    X_T = X.T

    XTX = X_T.dot(X)
    XTX = XTX + 0.000001 * np.eye(XTX.shape[0]) # function is not supposed to be regularized, but we're adding this to escape errors
    XTX_inverse = np.linalg.inv(XTX)

    w = XTX_inverse.dot(X_T).dot(y)

    return w[0], w[1:]

print("train_linear_regression(X, y): ", train_linear_regression(X, y))

# Section: Baseline Model - applying LR function to real datasets

base = ["engine_hp", "engine_cylinders", "highway_mpg", "city_mpg", "popularity"] # only use number cols
max_year = df.year.max()

# implementing categorical values

categorical_columns = [
    'make', 'engine_fuel_type', 'transmission_type', 'driven_wheels',
    'market_category', 'vehicle_size', 'vehicle_style'
]

categories = {}

for c in categorical_columns:
    categories[c] = list(df[c].value_counts().iloc[:10].index)

def prepare_X(df):
    df = df.copy()
    features = base.copy()

    # NB: for "year" we use feature engineering and create an "age" column
    # adding "age" reduces rmse by 0.2 approx

    df["age"] = max_year - df.year
    features.append("age")

    for n in [2, 3, 4]:
        df["num_doors_%s" % n] = (df["number_of_doors"] == n).astype(int)
        features.append("num_doors_%s" % n)

    for c, values in categories.items():
        for v in values:
            name = "%s_%s" % (c, v)
            df[name] = (df[c] == v).astype(int)
            features.append(name)

    X = df[features].fillna(0).values # fill missing with 0's - ok for models (they just ignore missing), sometimes better to fill with mean values
    return X

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)

def show_y_comparison(y, y_pred):
    sns.histplot(y_pred, bins=50, alpha=0.5, color='red')
    sns.histplot(y, bins=50, alpha=0.5, color='green')
    plt.show()

# show_y_comparison(y_train, y_pred) # shows that average prediction is lower than real price (lower bell curve) & that at some places there are significant differences

# --> found the base model from which to improve

# Section: RMSE - Root Mean Square Error

# to evaluate the model we calculate rmse(y_train, y_pred), where y_pred is the result of model & y_train - actual values
# 1st we calculate error: y_train[i] - y_pred[i]
# 2nd is get squared error: (y_train[i] - y_pred[i])**2 # square & then root is to get "absolute" error disregarding whether it's negative or positive
# 3rd is get mean over all vector members: (se1, se2, ... sen) / n
# 4th is get a root from mse

def rmse(y, y_pred):
    e = y - y_pred
    se = e ** 2
    mse = se.mean()
    rmse = np.sqrt(mse)

    return rmse

print("rmse (train): ", rmse(y_train, y_pred))

# Section: Validating the Model - calculating rmse for validation set

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

# show_y_comparison(y_val, y_pred)

print("rmse (val): ", rmse(y_val, y_pred))

# Section: Categorical Features

# the way to go abt categorical columns is to create several additional binary features - each corresponding to a value from a set
# see prepare_X function
# NB: even numerical columns can be categorical (number_of_doors) - the numbers might not correspond with the target in a quantitative way

# Section: Regularization

# NB: using matrix inversion for finding (w) can give bad results in case of singular or nearly-singular feature matrices
# matrices that have columns representable as linear combinations of others do not have inverses - they lose information on transformations as they "flatten" dimensions
# nearly-singular matrices (singular with noise) give large numbers in their inverses when trying to retrieve info from points "flattened" into noise

# here is nearly-singular case - noise prevents errors but gives huge weights
# solution - regularization - adding a parameter to the diagonal (0.01 for example)

def train_linear_regression_reg(X, y, r=0.01):
    X = np.column_stack([np.ones(X.shape[0]), X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inverse = np.linalg.inv(XTX)

    w = XTX_inverse.dot(X.T).dot(y)

    return w[0], w[1:]

w0, w = train_linear_regression_reg(X_train, y_train)

y_pred = w0 + X_val.dot(w)
print("rmse (val, regularized): ", rmse(y_val, y_pred))

# Section: Tuning the model - selecting best regularization parameter (r)

# we will do it by simply trying several and seeing which performs best
def compare_rs():
    for r in [0.0, 0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]:
        w0, w = train_linear_regression_reg(X_train, y_train, r=r)
        y_pred = w0 + X_val.dot(w)
        score = rmse(y_val, y_pred)

        print(r, w0.round(2), score)

# compare_rs()

# --> this showed that r~0.0001 is a good parameter with only slightest worsening as it increases (+ 0.02 rmse as it grows to 10)

# Section: Using the model - training with train+val & using on a simple row

X_full_train = prepare_X(df_full_train)
X_test = prepare_X(df_test)
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

y_pred = w0 + X_test.dot(w)
print("rmse (test): ", rmse(y_test, y_pred))

show_y_comparison(y_test, y_pred)

# using the model

case_index = 620
test_case = df_test.iloc[case_index].to_dict() # pretending we got a dictionary from API

def use_model(case, index=0):
    df_use = pd.DataFrame([case])
    X_use = prepare_X(df_use)

    y_pred = w0 + X_use.dot(w)

    print()
    print("----------- results %s %s ------------" % (case["make"], case["model"]))
    print()
    print(case)
    print()

    print("pred price: ", np.expm1(y_pred[0]))
    if index:
        print("real price: ", np.expm1(y_test[index]))

use_model(test_case, index=case_index)

golf = {
    'make': 'volkswagen',
    'model': 'golf',
    'year': 2005,
    'engine_fuel_type': 'diesel',
    'engine_hp': 115.0,
    'engine_cylinders': 4.0,
    'transmission_type': 'manual',
    'driven_wheels': 'front_wheel_drive',
    'number_of_doors': 4.0,
    'market_category': 'hatchback,diesel',
    'vehicle_size': 'compact',
    'vehicle_style': '4dr_hatchback',
    'highway_mpg': 34,
    'city_mpg': 24,
    'popularity': 1500
}

use_model(golf)

