from flask import Flask, request, jsonify
import pickle
from model_files.ml_model import predict_mpg

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    vehicle = request.get_json()
    print(vehicle)
    with open('./model_files/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    predictions = predict_mpg(vehicle, model)

    result = {
        'mpg_prediction' : list(predictions)
    }
    return jsonify(result)


@app.route('/test', methods=['GET'])
def test():
    return 'Pinging model application!'

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)