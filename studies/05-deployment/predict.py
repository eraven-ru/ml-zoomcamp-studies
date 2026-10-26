import pickle
from flask import Flask, request, jsonify

# use model from binary

input_file = 'model_C=1.0.bin'

def print_section(section):
    print('\n---------- %s -----------\n' %section)

with open(input_file, 'rb') as in_f:
    (model, dv) = pickle.load(in_f)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    print_section('model results')
    print('churn probability: ', y_pred)

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(y_pred >= 0.5),
    }

    return jsonify(result)

@app.route('/ping', methods=['GET'])
def ping():
    return 'pong (churn)'

print()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)