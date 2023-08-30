import os
from deployment_model import DeploymentModel

from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'images')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

mobdjbf = DeploymentModel(dvgdvf)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        return 'File uploaded successfully'


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory('static/images', filename)


if __name__ == '__main__':
    app.run(debug=True)
