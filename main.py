# import os
# from PIL import Image
# from flask import Flask, render_template, request, redirect, url_for, send_from_directory
# from werkzeug.utils import secure_filename
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# # Flask app initialization
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads/'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# # Define the paths for training and validation directories (assuming these exist)
# BASE_DIR = "C:\\Users\\V\\Desktop\\MLproject"
# TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
# VALID_DIR = os.path.join(BASE_DIR, "data", "validation")

# # Model path
# MODEL_PATH = os.path.join(BASE_DIR, "model", "image_classifier_model.h5")

# # Verify the model path
# if not os.path.exists(MODEL_PATH):
#     print(f"Model file not found at {MODEL_PATH}")
#     exit(1)  # Exit if model is not found

# # Load the trained model
# model = load_model(MODEL_PATH)
# print("Model loaded successfully.")

# # Function to verify and clean up invalid/corrupted images
# def verify_images(directory):
#     for root, _, files in os.walk(directory):
#         for file in files:
#             file_path = os.path.join(root, file)
#             try:
#                 with Image.open(file_path) as img:
#                     img.verify()
#             except (IOError, SyntaxError) as e:
#                 print(f"Removing corrupted image: {file_path} - {e}")
#                 os.remove(file_path)

# # Verify training and validation directories
# verify_images(TRAIN_DIR)
# verify_images(VALID_DIR)

# # Function to check allowed file extensions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# # Home route
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Check if the file is part of the request
#         if 'file' not in request.files:
#             return redirect(request.url)

#         file = request.files['file']

#         # If no file is selected
#         if file.filename == '':
#             return redirect(request.url)

#         # Validate file type
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             # Predict the uploaded image
#             prediction = predict_image(filepath)

#             return render_template('index.html', filename=filename, prediction=prediction)

#     return render_template('index.html')

# # Prediction function
# def predict_image(filepath):
#     try:
#         # Load and preprocess the image
#         image = load_img(filepath, target_size=(150, 150))
#         image = img_to_array(image)
#         image = np.expand_dims(image, axis=0)
#         image /= 255.0

#         # Predict the class
#         prediction = model.predict(image)[0][0]

#         # Get class labels based on training generator
#         # Assuming 0 = Dog and 1 = Cat based on previous implementation
#         class_labels = {0: "Dog", 1: "Cat"}
#         predicted_label = class_labels[int(prediction >= 0.5)]

#         print(f"Prediction: {prediction}, Predicted Label: {predicted_label}")
#         return predicted_label

#     except Exception as e:
#         print(f"Prediction error: {type(e).__name__}: {e}")
#         return "Prediction Error"

# # Route to display uploaded image
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# # Main entry point
# if __name__ == '__main__':
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     app.run(debug=True)
import os
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Flask app initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the paths for training and validation directories
BASE_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VALID_DIR = os.path.join(BASE_DIR, "data", "validation")
MODEL_PATH = os.path.join(BASE_DIR, "model", "image_classifier_model.h5")

# Verify and create necessary directories
for directory in [TRAIN_DIR, VALID_DIR, os.path.join(BASE_DIR, "model")]:
    os.makedirs(directory, exist_ok=True)

# Verify the model path
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}")
    exit(1)

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Function to verify and clean up invalid/corrupted images
def verify_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                print(f"Removing corrupted image: {file_path} - {e}")
                os.remove(file_path)

# Verify training and validation directories
verify_images(TRAIN_DIR)
verify_images(VALID_DIR)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home route (GET and POST)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                prediction = predict_image(filepath)
                return render_template('index.html', filename=filename, prediction=prediction)
            except Exception as e:
                print(f"Error processing file: {e}")
                return "Error processing the file.", 500
    return render_template('index.html')

# Prediction function
def predict_image(filepath):
    try:
        image = load_img(filepath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.0
        prediction = model.predict(image)[0]
        class_labels = {0: "Cat", 1: "Dog", 2: "Unknown"}
        predicted_index = np.argmax(prediction)
        predicted_label = class_labels.get(predicted_index, "Unknown")
        return predicted_label
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction Error"

# Route to display uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route for developers page
@app.route('/developers')
def developers():
    return render_template('developers.html')

# Route for setting page
@app.route('/setting')
def setting():
    return render_template('setting.html')

# Route for about page
@app.route('/about')
def about():
    return render_template('about.html')

# Handle 404 errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
