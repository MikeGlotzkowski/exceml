import json
import logging
import os
import threading
from flask import Flask, request, jsonify, render_template, redirect, send_from_directory, url_for
import numpy as np
import pandas as pd
from autosklearn.estimators import AutoSklearnClassifier
from sklearn.model_selection import train_test_split
import threading
from flask import Flask, request, jsonify, render_template, redirect, send_from_directory, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
#import DataPreprocessorChoice 


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
    thread = threading.Thread(target=start_automl, args=[experiment, app.app_context()])
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

def start_automl(experiment, context ):

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

    with context:
        db.session.add(experiment)
        experiment.results = automl.cv_results_
        experiment.sprint_statistics = automl.sprint_statistics()
        experiment.leaderboard = automl.leaderboard()
        experiment.best = _best
        db.session.commit()
        app.logger.info('finished automl for experiment with id: {}'.format(experiment.id))
        return True
    return False

# create route to get leaderboard for experiment by id
@app.route('/get_leaderboard', methods=['POST'])
def get_leaderboard():
    id = request.form['id']
    experiment = Experiment.query.get(id)
    leaderboard = pd.DataFrame(experiment.leaderboard)
    return leaderboard.to_json()

# create route to get sprint_statistics for experiment by id
@app.route('/get_sprint_statistics', methods=['POST'])
def get_sprint_statistics():
    id = request.form['id']
    experiment = Experiment.query.get(id)
    sprint_statistics = experiment.sprint_statistics
    return jsonify(sprint_statistics)

# create route to get best model for experiment by id
@app.route('/get_best_model', methods=['POST'])
def get_best_model():
    id = request.form['id']
    experiment = Experiment.query.get(id)
    best_model = experiment.best
    return json.dumps(best_model, cls=NpEncoder)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, DataPreprocessorChoice):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# create Experiment class
class Experiment(db.Model):
    __tablename__ = 'experiments'
    id = db.Column(db.Integer, primary_key=True)
    x_df = db.Column(db.PickleType)
    y_df = db.Column(db.PickleType)
    y_column = db.Column(db.String)
    x_columns = db.Column(db.PickleType)
    columns = db.Column(db.PickleType)
    results = db.Column(db.PickleType)
    best = db.Column(db.PickleType)
    sprint_statistics = db.Column(db.PickleType)
    leaderboard = db.Column(db.PickleType)


    def __repr__(self):
        return '<Experiment {}>'.format(self.id)

    def to_dict(self):
        return {
            'id': self.id,
            'x_df': self.x_df,
            'y_df': self.y_df,
            'y_column': self.y_column,
            'x_columns': self.x_columns,
            'columns': self.columns,
            'results': self.results,
            'best': self.best,
            'sprint_statistics': self.sprint_statistics,
            'leaderboard': self.leaderboard
        }


def get_create_table_statement_for_experiment():
    return """
    CREATE TABLE IF NOT EXISTS experiments (
        id SERIAL PRIMARY KEY,
        x_df BYTEA,
        y_df BYTEA,
        y_column VARCHAR,
        x_columns BYTEA,
        columns BYTEA,
        results BYTEA,
        best BYTEA,
        sprint_statistics BYTEA,
        leaderboard BYTEA
    );
    """
        

def main():
    app.run(debug=True)
    logger = logging.getLogger('werkzeug')
    logger.setLevel(logging.INFO)
    app.logger.addHandler(logger)



if __name__ == '__main__':
    main()