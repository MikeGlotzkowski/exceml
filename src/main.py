from enum import auto
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from autosklearn.estimators import AutoSklearnClassifier
import sklearn.metrics

def start():
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=5, random_state=1)

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # define search
    automl = AutoSklearnClassifier(time_left_for_this_task=2*60, per_run_time_limit=30, n_jobs=-1, ensemble_kwargs = {'ensemble_size': 1})
    # perform the search
    automl.fit(X_train, y_train)
    # summarize
    print(automl.sprint_statistics())

    print(automl.leaderboard().sort_values('rank', ascending=True).head(10))


    _best = [model for model in automl.show_models().values() if model["rank"] == 1][0]
    print("PRINT BEST MODEL")
    print(_best)

    cv_results = automl.cv_results_
    # write to csv file
    import pandas as pd
    pd.DataFrame(cv_results).to_csv("./data/cv_results.csv", index=False)


def main():
    start()

if __name__ == "__main__":
    main()

