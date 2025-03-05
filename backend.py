from flask import Flask, request, jsonify ,render_template
from tensorflow.keras.models import load_model
from flask_cors import CORS
import matplotlib.pyplot as plt
import numpy as np
import cv2

app = Flask(__name__)
CORS(app, origins=["https://doodle-classifier-cnn-project.onrender.com"])

# Load the model
model = load_model('./modelD.h5')  # Update with the path to your model

# Define class labels
CLASS_LABELS =['The Eiffel Tower', 'airplane', 'alarm clock', 'ant', 'apple',
       'axe', 'backpack', 'banana', 'baseball', 'bathtub', 'bear', 'bee',
       'bicycle', 'bird', 'birthday cake', 'book', 'bridge', 'butterfly',
       'calculator', 'camera', 'candle', 'car', 'carrot', 'cat', 'circle',
       'cloud', 'cow', 'crab', 'cup', 'dog', 'dolphin', 'elephant',
       'envelope', 'eye', 'fan', 'foot', 'frog', 'grapes', 'grass',
       'guitar', 'hat', 'house', 'ice cream', 'key', 'keyboard', 'knife',
       'ladder', 'leaf', 'line', 'moon', 'mountain', 'panda', 'pillow',
       'rainbow', 'river']

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Make sure 'index.html' is inside the 'templates' folder


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    file = request.files['file']
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    img_inverted = cv2.bitwise_not(img)

    # Resize to the input shape of the model
    img = cv2.resize(img_inverted, (28, 28))  # Update with your input size
    img = img.reshape(1, 28, 28, 1) / 255.0  # Normalize and add batch dimension

    # Predict the class
    predictions = model.predict(img)
    class_id = np.argmax(predictions)
    print(class_id)
    class_label = CLASS_LABELS[class_id]

    return jsonify({'class': class_label, 'confidence': float(predictions[0][class_id])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)