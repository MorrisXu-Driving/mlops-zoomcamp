import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify

# Setting MLFLOW ENV VAR
RUN_ID = '716a1ded1c9d44bca420809f492f67aa'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000/'

# Download Dictvectorizer from S3bucket through tracking server
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
RUN_ID = client.search_runs(experiment_ids=['1'])[0].info.run_id
path = client.download_artifacts(run_id=RUN_ID, path='dict_vectorizer.bin')
with open(path, 'rb') as f_out:
    dv = pickle.load(f_out)

# Load model from s3 bucket through mlflow
logged_model = f's3://mlflow-artifacts-remote-morrisxu/1/716a1ded1c9d44bca420809f492f67aa/artifacts/model'
model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):
    features = {} 
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])


app = Flask('duration-prediction')
 

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
