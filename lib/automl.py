import os
import pandas as pd
import numpy as np
from lib.util import timeit, Config
from lib.read import read_df
from lib.preprocess import preprocess
from lib.model import train, predict, validate
from typing import Optional
import gc
import time

class AutoML:
    def __init__(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        self.config = Config(model_dir)

    def train(self, train_csv: str, mode: str):
        self.config["task"] = "train"
        self.config["mode"] = mode

        self.config["objective"] = "regression" if mode == "regression" else "binary"
        self.config["metric"] = "rmse" if mode == "regression" else "auc"

        self.config.tmp_dir = self.config.model_dir + "/tmp"
        os.makedirs(self.config.tmp_dir, exist_ok=True)

        df = read_df(train_csv, self.config)
        df = preprocess(df, self.config)

        y = df["target"].copy()
        X = df.drop("target", axis=1).copy()
        del df
        gc.collect()

        self.config["columns"] = list(X)

        train(X, y, self.config)

    def predict(self, test_csv: str, prediction_csv: str) -> (pd.DataFrame, Optional[np.float64]):
        self.config["task"] = "predict"
        self.config.tmp_dir = os.path.dirname(prediction_csv) + "/tmp"
        os.makedirs(self.config.tmp_dir, exist_ok=True)

        self.config["prediction_csv"] = prediction_csv
        self.config["line_id"] = []
        
        self.config["start_time"]  = time.time()

        result = {
            "line_id": [],
            "prediction": [],
        }

        X = pd.read_csv(
                test_csv,
                encoding="utf-8",
                low_memory=False,
                dtype=self.config["dtype"],
                parse_dates=self.config["parse_dates"],
        )
        self.config["line_id"] = X["line_id"].values

        result["line_id"] = (X["line_id"].values)
        X = preprocess(X, self.config)

        X = X[self.config["columns"]]  # for right columns order

        result["prediction"] = predict(X, self.config)

        result = pd.DataFrame(result)
        result.to_csv(prediction_csv, index=False)

        target_csv = test_csv.replace("test", "test-target")
        if os.path.exists(target_csv):
            score = validate(result, target_csv, self.config["mode"], self.config)
        else:
            score = None

        return result, score

    @timeit
    def save(self):
        self.config.save()

    @timeit
    def load(self):
        self.config.load()
