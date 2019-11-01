import os
from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

graph = tf.compat.v1.get_default_graph()
with graph.as_default():
    model = load_model(STATIC_FOLDER + '/' + 'model_baseline.h5')


def api(full_path):
    data = image.load_img(full_path, target_size=(150, 150, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    with graph.as_default():
        predicted = model.predict(data)
        return predicted


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=["POST", "GET"])
def upload_file():
    if request.method is "GET":
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        indices = {0: 'cat', 1: 'dog'}
        result = api(full_name)

        predicted_class = np.asscalar(np.argmax(result, axis=1))
        accuracy = round(result[0][predicted_class] * 100, 2)
        label = indices[predicted_class]

    return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
