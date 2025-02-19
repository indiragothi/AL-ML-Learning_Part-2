from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import cv2  # OpenCV for image resizing

# Initialize Flask app
app = Flask(__name__)

# Load the trained SVM model saved with joblib
model = load('svm_digit_recog_model.pkl')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read uploaded image file
    file = request.files['image']
    img = plt.imread(file)  # Read image as numpy array

    # Convert the image to grayscale (assuming the uploaded image is RGB)
    if img.ndim == 3:  # Check if the image has RGB channels
        img = np.mean(img, axis=2)  # Convert to grayscale (average of RGB)
    
    # Resize the image to 8x8 pixels using OpenCV
    img_resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    
    # Debugging: Print resized image shape and some pixel values
    print(f"Resized image shape: {img_resized.shape}")
    print(f"Sample pixels from resized image (after conversion): {img_resized[:2, :2]}")

    # Normalize pixel values to the range [0, 16]
    img_rescaled = (img_resized / 255.0) * 16
    
    # Debugging: Print some values after normalization
    print(f"Normalized values (scaled to 0-16): {img_rescaled[:2, :2]}")

    # Flatten the image for the model
    input_data = img_rescaled.reshape(1, -1)

    # Predict using the loaded model
    predicted_label = model.predict(input_data)[0]

    # Encode image for display
    buf = BytesIO()
    plt.imshow(img_resized, cmap='gray')
    plt.axis('off')
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')

    return jsonify({'digit': int(predicted_label), 'image': encoded_img})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
