from __future__ import print_function
import os
import re
import os.path
from os import listdir
from os.path import isfile
from os.path import join
import shutil
import numpy as np
import inspect
from flask import Flask
from flask import send_from_directory
from flask import request
from flask import render_template
from keras import regularizers
from flask import redirect
from flask import url_for
import keras.backend.tensorflow_backend as tb
from PIL import Image
from werkzeug.utils import secure_filename
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from keras_preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import *
from image_search_engine import image_search_engine
import matplotlib.pyplot as plt


tb._SYMBOLIC_SCOPE.value = True
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg'])
glove_model_path = "/Users/ganeshkumaranmasilamani/Desktop/glove"
data_path = "/Users/ganeshkumaranmasilamani/Desktop/imagesearch"
features_path = "/Users/ganeshkumaranmasilamani/Desktop/Image_features"
file_mapping_path = "/Users/ganeshkumaranmasilamani/Desktop/Image_mapping"
custom_features_path = "/Users/ganeshkumaranmasilamani/Desktop/custom_features"
custom_features_file_mapping_path = "/Users/ganeshkumaranmasilamani/Desktop/custom_mapping"

images, vectors, image_paths, word_vectors = load_images_vectors_paths(glove_model_path, data_path)
model = image_search_engine.load_headless_pretrained_model()
images_features, file_index = image_search_engine.load_features(features_path, file_mapping_path)
image_index = image_search_engine.index_features(images_features)

#setting up the website template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def clear_dir(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))

def searchInternal(image_path, model):
    fv = image_search_engine.get_feature_vector(model, image_path)
    results = image_search_engine.search_index_by_value(fv, image_index, file_index)
    return results

def getImagesFilePathsFromFolder(path):
    onlyfiles = [ join(path,f) for f in listdir(path) if ( isfile(join(path, f)) and (".jpg" in f) )]
    return onlyfiles

def search():
    fileCount = len(getImagesFilePathsFromFolder(UPLOAD_FOLDER)) 
    fig, ax = plt.subplots(1,fileCount, figsize=(50,50))
    img_Counter=0;
    output = {}
    for img_path in getImagesFilePathsFromFolder(UPLOAD_FOLDER):
        print(img_path)
        image_url = 'http://127.0.0.1:9000/uploads/' + img_path.split("/")[-1]
        print(image_url)
        results = searchInternal(img_path, model)
        for result in results:
            img_path = result[1]
            shutil.copy(img_path, UPLOAD_FOLDER)
            print("file " + img_path + " copied successfully.")
            image_url = 'http://127.0.0.1:9000/uploads/' + img_path.split("/")[-1]
            output[image_url] = "" 
    return output

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file():
    if request.method == 'POST':
        print("clearing upload dir")
        clear_dir(UPLOAD_FOLDER)
        print("done")
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            output = search()
            if not (output is None):
                for imageUrl,character in output.items():
                    print(imageUrl)
                    print(character)
            return render_template('GroupN-imagesearchengine.html', character=character ,output=output)
    else:
    	return render_template('GroupN-imagesearchengine.html', review="" ,output=None)


@app.route("/")
def hello():
	return TEMPLATE_DIR

@app.route('/uploads/<filename>/')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/upload",	 methods=['GET', 'POST'])
def rec():
	return upload_file()

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=9000, debug=False, threaded=False)
