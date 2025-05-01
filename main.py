from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tf_keras
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = tf_keras.models.load_model(
    "model/cat_dog_model.h5",
    custom_objects={'KerasLayer': hub.KerasLayer}
)

def predict_image(image_path):
    input_img = cv2.imread(image_path)
    input_img_resize = cv2.resize(input_img, (224, 224))
    input_img_scaled = input_img_resize / 255.0
    image_reshaped = np.reshape(input_img_scaled, [1, 224, 224, 3])
    predictions = model.predict(image_reshaped)
    label = np.argmax(predictions)
    return "Cat" if label == 0 else "Dog"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None

    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"
        file = request.files["image"]
        if file.filename == "":
            return "No selected file"
        if file:
            filename = str(uuid.uuid4()) + "_" + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            image_url = file_path

    return render_template("index.html", prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
