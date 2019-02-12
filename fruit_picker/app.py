from flask import Flask, render_template, request, flash, redirect
import os
from keras.models import load_model
from keras import backend as K
import cv2
import numpy as np
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

IMG_SIZE = 64
CLASSES = ['banana', 'apple', 'orange']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            os.remove(filepath)
            K.clear_session()
            return render_template('home.html', prediction=prediction)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(url):
    model = load_model('./models/fruitpicker_v2.h5')

    path = url
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.normalize(img, None, alpha=0, beta=1,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    img = np.array(img)
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    model_pred = model.predict([img])
    total = 0
    for i in model_pred:
        for j in i:
            total += j
    prediction = CLASSES[np.argmax(model_pred)]
    return prediction


@app.route('/')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
