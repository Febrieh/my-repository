import os
from flask import Flask, request, render_template, url_for, send_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import logging

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
model = load_model('models/garbage_classifier.keras')

# Class labels (adjust as needed)
class_labels = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None
    result = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            logging.info(f'File saved to {file_path}')
            result = classify_image(file_path)
            logging.info(f'Classification result: {result}')
            image_url = url_for('uploaded_file', filename=file.filename)

    return render_template('index.html', result=result, image_url=image_url)

def classify_image(img_path):
    # Set the image dimensions
    img_height, img_width = 180, 180
    
    # Load and preprocess the image
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Perform prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Determine the predicted class and confidence
    predicted_class = class_labels[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    logging.info(f'Predicted class: {predicted_class}, Confidence: {confidence:.2f}')
    return predicted_class, f"{confidence:.2f}"

if __name__ == '__main__':
    app.run(debug=True)
