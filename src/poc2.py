import json
import logging
import os
import threading
from flask import Flask, request, jsonify, render_template, redirect, send_from_directory, url_for
import pandas as pd
from autosklearn.estimators import AutoSklearnClassifier
from sklearn.model_selection import train_test_split
import threading
from flask import Flask, request, jsonify, render_template, redirect, send_from_directory, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/postgres'
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)

# create allowed extensions
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    # log filename
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# route to create empty experiment, creates a new entry in postgres db and returns id
@app.route('/create_experiment', methods=['GET'])
def create_experiment():
    # create new experiment
    new_experiment = Experiment()
    # add to db
    db.session.add(new_experiment)
    db.session.commit()
    # return id
    return jsonify({'id': new_experiment.id})

# route upload_csv_for_experiment where you upload a csv for a specific experiment and enter the y_column
# add the csv to the db
@app.route('/upload_csv_for_experiment', methods=['POST'])
def upload_csv_for_experiment():
    file = request.files.get('file')
    y_column = request.form.get('y_column')
    id = request.form.get('id')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        df = pd.read_csv(file_path)
        columns = list(df.columns)
        x_columns = [x for x in columns if x != y_column]
        x_df = df[x_columns]
        y_df = df[y_column]
        experiment = Experiment.query.get(id)
        experiment.x_df = x_df
        experiment.y_df = y_df
        experiment.y_column = y_column
        experiment.x_columns = x_columns
        experiment.columns = columns
        db.session.commit()
        return jsonify({'id': id})
    return jsonify({'error': 'bad request'}), 400
# route to get all experiments from postgres db
@app.route('/get_experiments', methods=['GET'])
def get_experiments():

    # create a table with all experiments and a status
    # "new" if no x_df and y_df are present
    # "ready" if x_df and y_df are present and "results" is empty
    # "finished" if x_df and y_df are present and "results" is not empty

    # get all experiments
    experiments = Experiment.query.all()

    # create a touple of experiment and status
    experiments_with_status = []
    for experiment in experiments:
        if experiment.x_df is None and experiment.y_df is None:
            experiments_with_status.append((experiment, 'new_no_csv_uploaded'))
        elif experiment.x_df is not None and experiment.y_df is not None and experiment.results is None:
            experiments_with_status.append((experiment, 'ready_csv_uploaded'))
        elif experiment.x_df is not None and experiment.y_df is not None and experiment.results is not None:
            experiments_with_status.append((experiment, 'finished_result_available'))
    
    experiments_with_status_as_json = []
    for experiment, status in experiments_with_status:
        experiments_with_status_as_json.append({
            'id': experiment.id,
            'status': status,
            'y_column': experiment.y_column,
            'x_columns': experiment.x_columns,
            'columns': experiment.columns,
            'results': experiment.results,
            'best': experiment.best
        })
    return jsonify(experiments_with_status_as_json)
    


# route to start experiment
# load csv for experiment from sql db and start automl
@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    id = request.form['id']
    experiment = Experiment.query.get(id)
    
    # create a new thread to start automl
    thread = threading.Thread(target=start_automl, args=[experiment])
    thread.start()
    
    return jsonify({'id': id})


# get experiment results
# load results from sql db for experiment
@app.route('/get_experiment_results', methods=['POST'])
def get_experiment_results():
    id = request.form['id']
    experiment = Experiment.query.get(id)
    results = experiment.results
    return jsonify(results)

def start_automl(experiment):

    app.logger.info('starting automl for experiment with id: {}'.format(experiment.id))

    x_df = experiment.x_df
    y_df = experiment.y_df
    y_column = experiment.y_column
    x_columns = experiment.x_columns
    # columns = experiment.columns
    x_df.columns = x_columns
    y_df.columns = [y_column]
    
    X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.33, random_state=1)
    # define search
    automl = AutoSklearnClassifier(time_left_for_this_task=2*60, per_run_time_limit=30, n_jobs=-1, ensemble_kwargs = {'ensemble_size': 1})
    # perform the search
    automl.fit(X_train, y_train)
    _best = [model for model in automl.show_models().values() if model["rank"] == 1][0]

    experiment.results = automl.cv_results_
    experiment.best = _best
    db.session.commit()

    app.logger.info('starting automl for experiment with id: {}'.format(experiment.id))

    return True


# create Experiment class
class Experiment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    x_df = db.Column(db.PickleType)
    y_df = db.Column(db.PickleType)
    y_column = db.Column(db.String(80))
    x_columns = db.Column(db.PickleType)
    columns = db.Column(db.PickleType)
    results = db.Column(db.PickleType)
    best = db.Column(db.PickleType)
    def to_dict(self):
        return {
            'id': self.id,
            'x_df': self.x_df,
            'y_df': self.y_df,
            'y_column': self.y_column,
            'x_columns': self.x_columns,
            'columns': self.columns,
            'results': self.results,
            'best': self.best
        }

def get_create_table_statement_for_experiment():
    return """
        CREATE TABLE experiment (
            id SERIAL PRIMARY KEY,
            x_df BYTEA,
            y_df BYTEA,
            y_column VARCHAR(80),
            x_columns BYTEA,
            columns BYTEA,
            results BYTEA,
            best BYTEA
        );
    """

def main():
    app.run(debug=True)
    logger = logging.getLogger('werkzeug')
    logger.setLevel(logging.INFO)
    app.logger.addHandler(logger)



if __name__ == '__main__':
    main()