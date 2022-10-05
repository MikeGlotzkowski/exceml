# create a web api that lets you upload an csv file
# then write the csv to mongo db

import os
import sys
import csv
import json
import threading
import pymongo
import datetime
import requests
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, send_from_directory, url_for
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
from bson.json_util import dumps
from enum import auto
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from autosklearn.estimators import AutoSklearnClassifier
import sklearn.metrics


# create flask app
app = Flask(__name__)

# create mongo client
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["poc1"]
collection = db["poc1"]

# create upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# create allowed extensions
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# create upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def format_model_as_html_table(model):
    html = "<table>"
    for key in model:
        html += f"<tr><td>{key}</td><td>{model[key]}</td></tr>"
    html += "</table>"
    return html

def start_automl(file_path, y_column):
    
    # read csv file
    df = pd.read_csv(file_path)
    
    y = df[y_column]
    X = df.drop(y_column, axis=1)
    
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # define search
    automl = AutoSklearnClassifier(time_left_for_this_task=2*60, per_run_time_limit=30, n_jobs=-1, ensemble_kwargs = {'ensemble_size': 1})
    # perform the search
    automl.fit(X_train, y_train)
    # summarize
    # print(automl.sprint_statistics())

    # print(automl.leaderboard().sort_values('rank', ascending=True).head(10))


    _best = [model for model in automl.show_models().values() if model["rank"] == 1][0]

    cv_results = automl.cv_results_
    # write to csv file
    now_in_milliseconds = datetime.datetime.now().timestamp() 
    pd.DataFrame(cv_results).to_csv(f"./data/cv_results_{now_in_milliseconds}.csv", index=False)

    # return html with _best
    # return format_model_as_html_table(_best)

# create automl route
@app.route('/automl', methods=['GET', 'POST'])
def upload_and_automl():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        y_column = request.form['y_column']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '' or y_column == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # create a new thread to run automl
            t = threading.Thread(target=start_automl, args=(file_path, y_column))
            t.start()
            
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    # return a upload form and add a hint to the y_column
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=text name=y_column placeholder="Choose target column (to predict)">
        <input type=submit value=Upload>
    </form>
    '''


    


# create uploaded file route
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # read csv file
    df = pd.read_csv('uploads/' + filename)
    # convert to json
    json_data = json.loads(df.to_json(orient='records'))
    # insert json to db
    collection.insert_many(json_data)
    #return 'File uploaded successfully' and add a picture
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <p>File uploaded successfully</p>
    <iframe src="https://giphy.com/embed/37nRXpCEP9H1f1WVrb" width="480" height="480" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>
    '''

    
# create a landing page with a link to the automl route and a link to the get data route with all options
@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <p><a href="/upload">Upload a file</a></p>
    <p><a href="/automl">Upload a file and run automl</a></p>
    <p><a href="/get_data_root">Get data</a></p>
    '''

# create a get data route with all options
@app.route('/get_data_root')
def get_data_root():
    # get all data from db
    data = collection.find()
    # convert to json
    json_data = json.loads(dumps(data))
    # return json
    return jsonify(json_data)

    

# create get route
@app.route('/get', methods=['GET'])
def get():
    # get all documents from db
    docs = collection.find()
    # convert to json
    json_data = json.loads(dumps(docs))
    # return json
    return jsonify(json_data)

# create route to get data as table by filename
@app.route('/get_data/<filename>')
def get_data(filename):
    # read csv file
    df = pd.read_csv('data/' + filename)

    # omit params column
    df = df.drop('params', axis=1)

    # sort table ascending by rank_test_scores
    df = df.sort_values('mean_test_score', ascending=False)

    # convert to json
    json_data = json.loads(df.to_json(orient='records'))
    # insert json to db
    collection.insert_many(json_data)
    # return as html table
    return df.to_html()

# create main function
def main():
    # run flask app
    app.run(debug=True)

# run main function
if __name__ == '__main__':
    main()




