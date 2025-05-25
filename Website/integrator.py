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
def predict():
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    img_data = data["image"].split(",")[1]  # Remove data:image/...;base64 prefix
    image_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_data = detector.findHands(img_data)
    lmlist = detector.findPosition(img_data, draw=False)

    # img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    # results = hands.process(img_rgb)

    if not img_data.multi_hand_landmarks:
        return jsonify({"prediction": "No hand detected"})

    landmarks = []
    for lm in lmlist.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    prediction = model.predict([landmarks])[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
