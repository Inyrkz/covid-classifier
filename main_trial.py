import os
import urllib.request
from flask import Flask, flash, request, redirect
from flask_ngrok import run_with_ngrok
from flask import url_for, render_template
from werkzeug.utils import secure_filename

# importing DL libraries
import base64
import numpy as np
from numpy import asarray
import io
from PIL import Image

import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Create folder for uploaded images
UPLOAD_FOLDER = '/content/drive/MyDrive/PneumoniaX-master/static/uploads'


app = Flask(__name__)
run_with_ngrok(app)
app.secret_key = 'iloveyou'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# configuring maximum upload size of a file in bytes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    '''Function to authenticate the file type'''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model():
    '''function to get the trained keras model'''
    global model
    # Model to detect Pneumonia
    # model = load_model("pneumonia_model.h5")
    # model = load_model("covid_0.989.h5")
    model = load_model("/content/drive/MyDrive/PneumoniaX-master/covid_0.989.h5")
    # Model to detect Covid
    # model2 = load_model("covid_model.h5")
    print(" * Models loaded!")


def preprocess_image(image, target_size):
    '''function to preprocess input image'''
    if image.mode != "RGB":
        image = image.convert("RGB")  # if image is not RGB then convert to RGB
    image = image.resize(target_size)  # resize image to defined target size
    # image = image.img_to_array(image)  # convert image to an array of numbers
    image = asarray(image)
    image = image.astype(np.float32)  # convert from uint8 to float32
    image = np.expand_dims(image, axis=0)  # expand the dimension of the image

    return image


print("   * Loading Keras Model...")
get_model()


# Pneumonia Route
@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumo():
    if request.method == 'GET':
            return render_template('pneumo.html')

    if request.method == 'POST':
        # if no file is uploaded    
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('pneumo'))
            # return redirect(request.url)

        file = request.files['file']

        # if file name is empty
        if file.filename == '':
            flash('No image selectd for uploading')
            return redirect(url_for('pneumo'))

        # if there is a file and the file format is right
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and Analyzed')

            # creating an image object from io.BytesIO
            file = Image.open(file)
            preprocessed_image = preprocess_image(file, target_size=(200, 200))

            prediction = model.predict_proba(preprocessed_image, verbose=0).tolist()
            # remember the model variable is global and has been called already
            # predict returns a numpy array with the predictions
            # .tolist() is used to convert the array to a python list
            # this is required for the jsonify call to be made later

            if prediction[0][0] == 1:
                result = 'Your Result is Normal'
            else:
                result = 'Pneumonia Detected'

            return render_template('result.html', filename=filename,
                                   prediction=result)
        else:
            flash('Allowed image types are ==> .png, .jpg, .jpeg')
            return redirect(url_for('pneumo'))


# Route for COVID 19 Detection
@app.route('/', methods=['GET', 'POST'])
def covid():
    if request.method == 'GET':
        return render_template('covid.html')

    if request.method == 'POST':
        # if no file is uploaded
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('covid'))

        file = request.files['file']

        # if file name is empty
        if file.filename == '':
            flash('No image selectd for uploading')
            return redirect(url_for('covid'))

        # if there is a file and the file format is right
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and Analyzed')

            # creating an image object from io.BytesIO
            file = Image.open(file)
            preprocessed_image = preprocess_image(file, target_size=(200, 200))
            # Change to model2 later ******************
            prediction = model.predict_proba(preprocessed_image, verbose=0).tolist()

            if prediction[0][0] == 1:
                result = 'Your Result in Normal'
            else:
                result = 'Covid Detected'

            return render_template('result.html', filename=filename,
                                   prediction=result)
        else:
            flash('Allowed image types are ==> .png, .jpg, .jpeg')
            return redirect(url_for('covid'))


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename),
                    code=301)


if __name__ == "__main__":
    app.run(debug=True)
