import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("./data.csv")
df = pd.DataFrame(data)

columns = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
target_column = "fuel_efficiency_mpg"

df = df[columns]

print(df.head())

def show_fuel_efficiency_distribution():
    sns.histplot(df["fuel_efficiency_mpg"])
    plt.show()

# show_fuel_efficiency_distribution()
# does not show long tail, pretty nice bell curve

# EDA

def show_df_eda():
    print("\n--------- EDA ---------\n")

    print("--> missing values\n")
    print(df.isnull().sum())

    print("\n--> unique values\n")
    print(df.nunique())

    print("\n--> horsepower stats\n")
    print(df.horsepower.describe())

    print("\n")


show_df_eda()
# 'horsepower' col has missing values
# 'horsepower' median is 254.000000

# Validation Framework

n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

idx = np.arange(n)

np.random.seed(42)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

y_train = df_train[target_column].values
y_val = df_val[target_column].values
y_test = df_test[target_column].values

del df_train[target_column]
del df_val[target_column]
del df_test[target_column]

def prepare_X(df, fill_mean=False):
    missing_swap = df_train.mean() if fill_mean else 0
    return df.fillna(missing_swap).values

def train_linear_regression(X, y):
    X = np.column_stack([np.ones(X.shape[0]), X])
    XTX = X.T.dot(X)

    XTX_inverse = np.linalg.inv(XTX)
    w = XTX_inverse.dot(X.T).dot(y)

    return w[0], w[1:]

def rmse(y, y_pred):
    e = y - y_pred
    se = e ** 2
    mse = se.mean()
    rmse_value = np.sqrt(mse)

    return rmse_value

X_train_z = prepare_X(df_train) # z for zeroes
X_train_m = prepare_X(df_train, fill_mean=True) # m for mean

X_val_z = prepare_X(df_val) # z for zeroes
X_val_m = prepare_X(df_val, fill_mean=True) # m for mean

w0_z, w_z = train_linear_regression(X_train_z, y_train)
w0_m, w_m = train_linear_regression(X_train_m, y_train)

y_pred_z = w0_z + X_val_z.dot(w_z)
y_pred_m = w0_m + X_val_m.dot(w_m)

print("--------- na zeroes vs mean comparison ---------\n")
print("rmse (na->zeroes): ", round(rmse(y_val, y_pred_z), 3)) # 2.205
print("rmse (na->mean): ", round(rmse(y_val, y_pred_m), 3)) # 2.202

# filling missing values with mean values gives slightly better rmse

# Regularization

def train_linear_regression_reg(X, y, r=0.001):
    X = np.column_stack([np.ones(X.shape[0]), X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inverse = np.linalg.inv(XTX)
    w = XTX_inverse.dot(X.T).dot(y)

    return w[0], w[1:]

X_train = prepare_X(df_train)
X_val = prepare_X(df_val)

def compare_r():
    r_options = [0, 0.01, 0.1, 1, 5, 10, 100]
    for r in r_options:
        w0, w = train_linear_regression_reg(X_train, y_train, r=r)
        y_pred = w0 + X_val.dot(w)

        print("r: ", r)
        print("rmse: ", rmse(y_val, y_pred), "\n")

print("\n-------- r comparison ----------\n")
compare_r()
# 0 gives the smallest rmse, 0.01 is alright - most probably means that we do not have even closely duplicate columns, so regularization is unnecessary (?)

# Seed Comparison

seed_scores = []

def append_seed_score(seed=1):
    idx = np.arange(n)

    np.random.seed(seed)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]

    y_train = df_train[target_column].values
    y_val = df_val[target_column].values
    y_test = df_test[target_column].values

    del df_train[target_column]
    del df_val[target_column]
    del df_test[target_column]

    X_train = prepare_X(df_train)
    X_val = prepare_X(df_val)
    w0, w = train_linear_regression(X_train, y_train)

    y_pred = w0 + X_val.dot(w)
    error = rmse(y_val, y_pred)
    seed_scores.append(error)

    print("\nseed: ", seed)
    print("score: ", round(error, 3))

print("\n----------- seed comparison -----------\n")

for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    append_seed_score(seed=s)

print("\nseed scores std: ", round(np.std(seed_scores), 3)) # 0.029

# Training with seed = 9

idx = np.arange(n)

np.random.seed(9)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]
df_full_train = pd.concat([df_train, df_val])

y_train = df_train[target_column].values
y_val = df_val[target_column].values
y_test = df_test[target_column].values
y_full_train = np.concatenate([y_train, y_val])

del df_train[target_column]
del df_val[target_column]
del df_test[target_column]
del df_full_train[target_column]

X_full_train = prepare_X(df_full_train)
X_test = prepare_X(df_test)

w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

y_pred = w0 + X_test.dot(w)
error = rmse(y_test, y_pred)

print("\n----------- seed = 9 test scoring -----------\n")
print("rmse (seed=9): ", round(error, 3)) # 2.236


