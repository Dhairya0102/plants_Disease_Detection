# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request
from PIL import Image
import io
import base64
import json

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
loaded_model = load_model('plant_disease_model.h5')

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

class_labels = class_names  # Use the same class names as before

# Home route
@app.route('/')
def home():
    return '''
    <h1>Plant Disease Detection</h1>
    <p>Upload an image of a plant leaf to detect its disease.</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file"><br><br>
      <input type="submit" value="Upload and Predict">
    </form>
    '''

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the file from POST request
    if 'file' not in request.files:
        return 'No file uploaded.', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected.', 400

    # Read the image
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    predicted_label = class_labels[predicted_class]

    # Prepare the image for display
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode('ascii')

    # Render the result
    return f'''
    <h1>Prediction Result</h1>
    <p><strong>Predicted Class:</strong> {predicted_label}</p>
    <p><strong>Confidence:</strong> {confidence:.2f}</p>
    <img src="data:image/jpeg;base64,{img_base64}" alt="Uploaded Image" width="300">
    <br><br>
    <a href="/">Back to Home</a>
    '''

# Run the app
if __name__ == '__main__':
    app.run(debug=True)