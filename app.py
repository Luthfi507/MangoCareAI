from flask import Flask, render_template, request
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

mango_classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge',
                 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# Set the upload folder
app.config["UPLOAD_FOLDER"] = "uploads"

# Load model
model = keras.models.load_model("model/vgg19-Mango Diseases-95.25.h5")

# Function to make predictions
def predict(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    prediction = model.predict(img)
    return prediction

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return render_template("index.html", message="No file part")

        file = request.files["file"]

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == "":
            return render_template("index.html", message="No selected file")

        if file:
            # Save the uploaded file to a temporary location
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)

            # Perform prediction
            prediction = predict(file_path)

            # You can process the prediction result here and return it to the user
            class_index = prediction.argmax()
            class_name = mango_classes[class_index]  # Replace with your class names
            result_message = f"The predicted class is {class_name} with confidence {prediction[0][class_index]}"

            return render_template("index.html", message=result_message)

    return render_template("index.html", message="Upload an image for prediction")

if __name__ == "__main__":
    app.run(debug=True)