import pandas as pd
import numpy as np
import time
from lib.automl import AutoML
from lib.util import timeit, log
import json

DATASETS = [
    ("1", "regression", 300),
    ("2", "regression", 300),
    ("3", "regression", 300),
    ("4", "classification", 300),
    ("5", "classification", 300),
    ("6", "classification", 600),
    ("7", "classification", 1200),
    ("8", "classification", 1800),
]

def validate_dataset(alias: str, mode: str, train_limit: int) -> np.float64:

    start_time = time.time()

    log(alias)

    automl = AutoML("models/check_{}".format(alias))

    automl.config["time_limit"] = train_limit
    # automl.load()

    automl.train("data/check_{}/train.csv".format(alias), mode)

    score_train_val = None
    if 'leak' not in automl.config:

        config = automl.config
        if config['mode']=='regression':
            best_oof = np.min([i['score_oof'] for i in config["lgb_cv_models"]])
        else:
            best_oof = np.max([i['score_oof'] for i in config["lgb_cv_models"]])

    out_log = pd.DataFrame(automl.config['log'])
    out_log.to_csv(f'models/check_{alias}/log_train.csv')
    print(pd.DataFrame(automl.config['log']))    

    end_time = time.time()
    train_time = end_time - start_time
    start_time = end_time


    automl.config["time_limit"] = 300
    _, score_test_val = automl.predict("data/check_{}/test.csv".format(alias), "predictions/check_{}.csv".format(alias))

    out_log = pd.DataFrame(automl.config['log'])
    out_log.to_csv(f'models/check_{alias}/log_test.csv')
    print(pd.DataFrame(automl.config['log']))

    end_time = time.time()
    test_time = end_time - start_time

    return best_oof, score_test_val, train_time, test_time

if __name__ == '__main__':
    scores = {
        "dataset": [],
        "score_train": [],
        "score_test": [],
        "train_time": [],
        "test_time": [],
    }

    for i, mode, train_limit in DATASETS:
        alias = "{}_{}".format(i, mode[0])

        start_time = time.time()
        score_train_val, score_test_val, train_time,test_time = validate_dataset(alias, mode,train_limit)

        end_time = time.time()

        scores["dataset"].append(alias)
        scores["score_train"].append(score_train_val)
        scores["score_test"].append(score_test_val)
        scores["train_time"].append(train_time)
        scores["test_time"].append(test_time)

    scores = pd.DataFrame(scores)
    scores.index.name = 'ix'
    scores.to_csv("scores/{}.csv".format(int(time.time())))
    print(scores)
