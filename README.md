# PneumoniaX
Building a flask app using deep learning to detect Covid 19 and Pneumonia from Chest X-Ray Images of Patients

The `CXR.ipynb` notebook is contains code for training a Convolutional Neural Network to detect Viral Pneumonia, COVID-19 and Normal Chest Radiographs.

The `CovidTrial.ipynb` notebook is contains code for training a Convolutional Neural Network to detect COVID-19 and Normal Chest Radiographs.

The `pneuX_classifier.ipynb` notebook is contains code for training a Convolutional Neural Network to detect Pneumonia and Normal Chest Radiographs.


Create a virtual environment by running the code below.

```bash
pip install virtualenv
virtualenv venv
```

Activate the virtual environment on Windows operating system by running the code below.

```
venv\Scripts\activate.bat
```

Install the libraries in the `requirements.txt` file by running the code below.

```bash
pip install -r requirements.txt
```

The deep learning model is too large to be stored on Github. 
You'll have to retrain the model by running the codes in the `CovidTrial.ipynb` and `pneuX_classifier.ipynb` notebook.

In the `app.py` file, change the code in `line 33` to the name of your deep learning model file.

`    model = load_model("model_0.886.h5")`

You can run the web application by running the code below in your terminal.

```python 
python app.py
```


