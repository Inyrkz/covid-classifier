'''
    Flask api to receive an image as input, run a DL model on it and get
    a JSON object as output
'''

# importing relevant libraries
import base64
from flask import Flask, request, jsonify
import numpy as np
from numpy import asarray
import io
from PIL import Image

import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# from tensorflow.keras.layers.experimental import preprocessing
# from keras.preprocessing.Image import ImageDataGenerator
# from keras.preprocessing.Image import img_to_array


# creating an instance of flask
app = Flask(__name__)


def get_model():
    '''function to get the trained keras model'''
    global model
    model = load_model("model_0.886.h5")
    print(" * Model loaded!")


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


# defining endpoints and http method
@app.route("/predict", methods=["POST", "GET"])
def predict():
    '''function to make predictions'''
    data = request.files['image']
    # data = request.files['image']
    # gets image file from client
    # encoded = data['image']
    # decoded = base64.b64decode(data)  # decoded image data
    # image = Image.open(io.BytesIO(decoded))
    # Image.open opens an image file.
    # The image data is in memory as bytes stored within the decoded variable.
    # It is not stored in an actual file, we need to wrap the bytes
    # in io.BytesIO and pass it to Image.open
    # Basically we are creating an image object
    img = Image.open(data.stream)
    preprocessed_image = preprocess_image(img, target_size=(200, 200))
    # preprocessing image with the function created earlier

    prediction = model.predict_proba(preprocessed_image, verbose=0).tolist()
    # remember the model variable is global and has been called already
    # predict returns a numpy array with the predictions
    # .tolist() is used to convert the array to a python list
    # this is required for the jsonify call to be made later

    response = {
        'prediction': {
            'Pneumonia': prediction[0][0],
            'No Pneumonia': prediction[0][1],
            'full': str(prediction)
                     }
    }
    # The response variable is defined as a python dictionary
    # It is the response we plan to send back to the client
    # with the naira currency prediction for the original image
    # It has a key 'prediction', which is also a dictionary. prediction
    return jsonify(response)

app.run(port=1234, debug=True)
