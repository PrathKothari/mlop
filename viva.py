from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

app = Flask(__name__)
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(X, y)

app= Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict(['data'])
    return jsonify({'prediction': int(prediction[0])})