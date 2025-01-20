from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
model = load_model('deepfakeclassifier.h5')

# Set upload folder and allowed extensions
UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess the image
def preprocess_image(img_path, scale_factor=0.5):
    img = cv2.imread(img_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    small_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        x, y, w, h = int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face.astype('float32') / 255.0
        return np.expand_dims(face, axis=0), img
    return None, None

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file part.")
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error="No selected file.")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image and predict
            processed_image, original_image = preprocess_image(file_path)
            if processed_image is None:
                return render_template('upload.html', error="No face detected in the image.")
            
            prediction = model.predict(processed_image)
            if prediction[0] > 0.5:
                result = f"Fake (Confidence: {prediction[0][0]:.2f})"
            else:
                result = f"Real (Confidence: {1 - prediction[0][0]:.2f})"
            
            return render_template('result.html', result=result, image_url=file_path)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
