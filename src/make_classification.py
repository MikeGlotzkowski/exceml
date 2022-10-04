
import pandas as pd
from sklearn.datasets import make_classification


def make_classification_df():
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=5, random_state=1)
    df = pd.DataFrame(X)
    df['y'] = y
    return df

def write_to_file(df):
    df.to_csv('./data/classification_example.csv', index=False)

def main():
    classification = make_classification_df()
    write_to_file(classification)
if __name__ == '__main__':
    main()