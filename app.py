from flask import Flask, render_template, request
import os
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import tensorflow_hub as hub
import tf_keras
from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
import datetime
import base64
import dotenv

dotenv.load_dotenv()

app = Flask(__name__)

imagekit = ImageKit(
    public_key=os.getenv("IMAGEKIT_PUBLIC_KEY"),
    private_key=os.getenv("IMAGEKIT_PRIVATE_KEY"),
    url_endpoint=os.getenv("IMAGEKIT_URL_ENDPOINT")
)

# Load model
model = tf_keras.models.load_model(
    "model/cat_dog_model.h5",
    custom_objects={'KerasLayer': hub.KerasLayer}
)

def predict_image_from_url(image_url):
    response = requests.get(image_url)
    if response.status_code != 200:
        return "Error: Unable to fetch image from URL."
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((224, 224))
    input_array = np.array(img) / 255.0
    image_reshaped = np.reshape(input_array, [1, 224, 224, 3])
    predictions = model.predict(image_reshaped)
    label = np.argmax(predictions)
    return "Cat" if label == 0 else "Dog"

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None

    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]
            if file.filename != "":
                file_extension = os.path.splitext(file.filename)[1]
                file_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + file_extension
                upload = imagekit.upload_file(
                    file=base64.b64encode(file.read()),
                    file_name=file_name,
                    options=UploadFileRequestOptions(
                        folder="/MeowBark"
                    )
                )
                image_url = upload.response_metadata.raw.get("url")
                prediction = predict_image_from_url(image_url)

        elif "image_url" in request.form:
            image_url = request.form["image_url"]
            prediction = predict_image_from_url(image_url)

    return render_template("index.html", prediction=prediction, image_url=image_url)


if __name__ == "__main__":
    app.run(debug=True)