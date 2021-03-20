import argparse

import numpy as np
from sklearn.linear_model import RidgeClassifier

import rocket


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name')
    parser.add_argument('--datasets-root', default='UCR_TS_Archive_2015')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset_name

    test_df = np.genfromtxt(f"{args.datasets_root}/{dataset_name}/{dataset_name}_TEST", delimiter=',')
    train_df = np.genfromtxt(f"{args.datasets_root}/{dataset_name}/{dataset_name}_TRAIN", delimiter=',')

    X_train, y_train = train_df[:, 1:], train_df[:, 0]
    X_test, y_test = test_df[:, 1:], test_df[:, 0]

    ts_length = X_train.shape[-1]

    classifier = RidgeClassifier(normalize=True)
    r = rocket.Rocket(input_length=ts_length, classifier=classifier)

    r.fit(X_train, y_train)

    accuracy = r.score(X_test, y_test)

    print(f"Dataset {dataset_name}, Test accuracy: {accuracy}")


if __name__ == '__main__':
    main()