from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# Load the trained model
model = tf.keras.models.load_model('models/model.h5')

# Initialize Flask app
app = Flask(__name__)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Define vegetable classes (update based on your dataset)
CLASSES = ['Carrot', 'Potato', 'Tomato', 'Cucumber', 'Onion']  # Change based on your dataset

# Set up folder for storing uploaded files
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Ensure a file is uploaded
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image
            img = cv2.imread(file_path)
            img = cv2.resize(img, (150, 150))  # Resize to match the model input size
            img = img / 255.0  # Normalize the image
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Predict the class
            result = model.predict(img)
            predicted_class = CLASSES[np.argmax(result)]

            # Set the prediction message
            prediction = f'Predicted Vegetable: {predicted_class}'

            # Optionally, you could remove the file after prediction to keep the server clean
            os.remove(file_path)

        else:
            prediction = 'Invalid file type. Please upload a valid image (jpg, jpeg, png).'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
# app.py