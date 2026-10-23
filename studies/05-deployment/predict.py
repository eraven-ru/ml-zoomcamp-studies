import pickle

# use model from binary

input_file = 'model_C=1.0.bin'

def print_section(section):
    print('\n---------- %s -----------\n' %section)

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

with open(input_file, 'rb') as in_f:
    (model, dv) = pickle.load(in_f)

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]

print_section('model results')
print('churn probability: ', y_pred)

print()