import time

import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.python.keras.models import model_from_json
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
data = ["반팔", "바지", "긴팔", "원피스", "팔이 긴 옷", "끈 신발", "셔츠", "운동화", "가방", "목있는 신발"]

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/classification/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        name, ext = os.path.splitext(f.filename)
        if ext != ".png":
            return "<script>alert('png파일만 업로드 가능합니다!'); window.location.href='/'</script>"
        name = name + str(time.time()) + ext
        f.save(secure_filename(name))
        json_file = open("model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")

        image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        image = image.reshape(-1, 28, 28, 1)
        image = image.astype('float32') / 255.0

        pred = loaded_model.predict(image)
        pred = np.argmax(pred, axis=1)

        return render_template('result.html', result=data[pred[0]])


if __name__ == '__main__':
    app.run()
