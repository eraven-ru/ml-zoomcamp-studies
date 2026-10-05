import pandas as pd

data = [
    ['Nissan', 'Stanza', 1991, 138, 4, 'MANUAL', 'sedan', 2000],
    ['Hyundai', 'Sonata', 2017, None, 4, 'AUTOMATIC', 'Sedan', 27150],
    ['Lotus', 'Elise', 2010, 218, 4, 'MANUAL', 'convertible', 54990],
    ['GMC', 'Acadia',  2017, 194, 4, 'AUTOMATIC', '4dr SUV', 34450],
    ['Nissan', 'Frontier', 2017, 261, 6, 'MANUAL', 'Pickup', 32340],
]

columns = [
    'Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders',
    'Transmission Type', 'Vehicle_Style', 'MSRP'
]

dict_data = [
    {
        "Make": "Nissan",
        "Model": "Stanza",
        "Year": 1991,
        "Engine HP": 138.0,
        "Engine Cylinders": 4,
        "Transmission Type": "MANUAL",
        "Vehicle_Style": "sedan",
        "MSRP": 2000
    },
    {
        "Make": "Hyundai",
        "Model": "Sonata",
        "Year": 2017,
        "Engine HP": None,
        "Engine Cylinders": 4,
        "Transmission Type": "AUTOMATIC",
        "Vehicle_Style": "Sedan",
        "MSRP": 27150
    },
    # ...
]

# create DataFrames from lists & dictionary; access first 5 and 3 rows;

df = pd.DataFrame(data, columns=columns)
dict_df = pd.DataFrame(dict_data)

print("df ", df)
print("dict_df ", dict_df)

print("df.head ", df.head())
print("df.head(n = 2) ", df.head(n=2))

# access 1 Series in 2 ways; access several Series simultaneously;

print("df.Make")
print(df.Make)

print('df["Engine HP"]')
print(df["Engine HP"])

print('df[["Make", "MSRP"]]')
print(df[["Make", "MSRP"]])

# create and delete a Series;

df["id"] = [67, 89, 99, 0.2, 77]
print("df with id")
print(df)

del df["id"]
print("df w/o id")
print(df)

# access rows by index; change index to letters; access by positional index; reset indices;

print("df.loc(2)")
print(df.loc[2])

df.index = ["g", 'h', 'i', 'j', 'k']
print("df with letter index")
print(df)
print(df.loc["j"])
print(df.iloc[1])

df = df.reset_index(drop=True)
print("df with base index")
print(df)

# perform math operations on Series;

print("df.MSRP - 200")
print(df.MSRP - 200)

# filter rows - get rows with prices bigger than 30000;

print("df[df.MSRP >= 30000]")
print(df[df.MSRP >= 30000])

# normalize Vehicle_Style Series;

df["Vehicle_Style"] = df["Vehicle_Style"].str.lower().str.replace(" ", "-")
print("normal Vehicle Style")
print(df)

# use summarizing operations & print all valuable stats for S and DF;

print("df.MSRP.min()")
print(df.MSRP.min())

print('df["Engine HP"].describe()')
print(df["Engine HP"].describe())

print("df.describe().round(2)")
print(df.describe().round(2))

# calculate all unique values in a Series and DF;

print('df["Transmission Type"].nunique()')
print(df["Transmission Type"].nunique())

print("df.nunique()")
print(df.nunique())

# check is values in DF exist; calculate null values for each S;

print("df.isnull().sum()")
print(df.isnull().sum())

# get mean price value for each manufacturer;

print('df.groupby("Make").MSRP.mean()')
print(df.groupby("Make").MSRP.mean())

# transform a Series into numpy array;

print("df.Year.values")
print(df.Year.values)

# transform a DF back to dictionary;

print("df.to_dict(orient='records')")
print(df.to_dict(orient='records'))