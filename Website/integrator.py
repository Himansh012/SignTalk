from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import joblib
import mediapipe as mp
import HandTrackingModule as htm

# Load the model
model = joblib.load("asl_model1.pkl")

# Setup hand tracking
cap = cv2.VideoCapture(0)
detector = htm.HandDectector()


app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    if 'frame' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if not lmList or len(lmList) < 21:
        return jsonify({"prediction": "No hand detected"})

    # Extract only x, y for model input (assuming model trained on 21 landmarks * 2 coords)
    flat_landmarks = []
    for lm in lmList:
        flat_landmarks.extend([lm[1], lm[2]])

    prediction = model.predict([flat_landmarks])[0]
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
