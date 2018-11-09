import pandas as pd
import numpy as np
import lightgbm as lgb
import hyperopt
from hyperopt import hp, tpe, STATUS_OK, STATUS_FAIL, space_eval, Trials
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from lib.util import timeit, log, Config
from typing import List, Dict
import time
import gc
import copy
from multiprocessing.pool import ThreadPool,Pool
from functools import partial
from sklearn.linear_model import ElasticNet, Lasso,\
BayesianRidge, LassoLarsIC, Ridge,LogisticRegression, RidgeCV,LogisticRegressionCV,LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler

@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    train_lightgbm(X, y, 
    stored_models_key='lgb_cv_models', save_to_disk=True, config=config)    

@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    return predict_lightgbm(X, config)

@timeit
def validate(preds: pd.DataFrame, target_csv: str, mode: str, config: Config) -> np.float64:
    df = pd.merge(preds, pd.read_csv(target_csv), on="line_id", left_index=True)
    score = roc_auc_score(df.target.values, df.prediction.values) if mode == "classification" else \
        np.sqrt(mean_squared_error(df.target.values, df.prediction.values))
    log("Score: {:0.4f}".format(score))
    return score

@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, stored_models_key:str, save_to_disk:bool, config: Config):

    config[stored_models_key] = []
    
    data = lgb.Dataset(X, label=y, free_raw_data=False)
    data.construct()
    gc.collect()

    params = {
        "objective": config["objective"],
        "metric": config["metric"],
        "seed": config["seed"],
        'num_threads':config['n_threads'],
        "verbosity": -1,
    }

    seed = config["seed"]

    space = {
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.4),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6, 10]),
        "num_leaves": hp.choice("num_leaves", np.linspace(4, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.1, 1., 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.1, 1., 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 20, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 30),
        "reg_lambda": hp.uniform("reg_lambda", 0, 30),
        "min_child_weight": hp.uniform('min_child_weight', 1e-10, 20),
        "max_bin": hp.choice('max_bin', [50,100,255]),
        'boosting_type': hp.choice('boosting_type', 
                                [
                                 {'boosting_type': 'gbdt', }, 
                                 {'boosting_type': 'dart', 
                                  'drop_rate': hp.uniform('drop_rate', 0.01, 0.6),
                                  'max_drop':  hp.choice("max_drop", np.linspace(5, config["train_num_boost_round"]*.9, 10, dtype=int)),
                                  'skip_drop': hp.uniform('skip_drop', 0.1, 0.7),
                                  },
                                #  {'boosting_type': 'rf', 
                                #   'bagging_freq': 1,
                                #   },         
                                #  {'boosting_type': 'goss', 
                                #   'bagging_freq': 0,
                                #   },                                  
                                ]),
        #train params
        'early_stopping_rounds': hp.choice("early_stopping_rounds", [None, 50]),
        'cv_splits': hp.choice("cv_splits", np.linspace(3, 12, 10, dtype=int)), # [4,8]
        'shuffle': hp.choice("shuffle", [True,False]),
    } 

    if config.is_classification():
        space['scale_pos_weight'] = hp.uniform('scale_pos_weight', 0.5, 10)
    else:
        space['objective'] = hp.choice("objective", ['regression',
                                                    'huber',
                                                    # 'fair',
                                                    # 'regression_l1',
                                                    ])
    def objective(space_sample):

        iteration_start = time.time()
        hyperparams = copy.deepcopy(space_sample)
        boosting_type = {}
        if 'boosting_type' in hyperparams.keys():
            boosting_type = hyperparams.pop('boosting_type')
        
        hyperparams = {**params, **hyperparams,**boosting_type}

        scores, models, y_oof = train_lightgbm_cv(data=data, 
                                            hyperparams=hyperparams, 
                                            config=config)

        if config.is_classification(): scores['oof'] = -scores['oof']

        iteration_time = time.time()-iteration_start
        log('iteration time %.1f, loss %.5f' % (iteration_time, scores['oof']))

        elapsed_time = (time.time() - config['start_time']) 
        have_time = (config["time_limit"] - elapsed_time - iteration_time) > 25
        if have_time:
            save_model(models, hyperparams, 
                    scores, y_oof, stored_models_key, save_to_disk, config)

            status = STATUS_OK
        else:
            status = STATUS_FAIL

        return {'loss': scores['oof'], 
                'runtime': iteration_time,
                'scores': scores,
                'models': models,
                'y_oof': y_oof,
                'status': status}

    have_time =True
    eval_n = 0
    trials = Trials()

    while have_time:
        iteration_start = time.time()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials, algo=tpe.suggest,
                            max_evals=eval_n+1, verbose=1,
                            rstate=np.random.RandomState(eval_n)) #TODO: (bug) if seed the same - in some cases it samples same values forever
        iteration_time = time.time()-iteration_start
        elapsed_time = (time.time() - config['start_time']) 
        have_time = (config["time_limit"] - elapsed_time - iteration_time) > 25
        eval_n+=1

def train_lightgbm_cv(data, hyperparams, config):

    num_boost_round = int(config["train_num_boost_round"])
    seed = hyperparams['seed']

    hyperparams = copy.deepcopy(hyperparams)
    early_stopping_rounds = hyperparams.pop("early_stopping_rounds")
    cv_splits = hyperparams.pop('cv_splits')
    shuffle = hyperparams.pop('shuffle')

    kf = KFold(n_splits=cv_splits, shuffle=shuffle, random_state=seed)

    y_train = config["target_data"]
    y_oof   = pd.Series(index=y_train.index)

    fold_scores = []
    models = []

    for n_fold, (train_index, val_index) in enumerate(kf.split(y_train)):

        train_data  = data.subset(train_index)
        valid_data  = data.subset(val_index)
        valid_sets  = (valid_data, train_data)
        
        model = lgb.train(
            params=hyperparams, train_set=train_data, valid_sets=valid_sets, 
            num_boost_round=num_boost_round, 
            early_stopping_rounds=early_stopping_rounds, 
            verbose_eval=config['verbose_eval'])

        score_tr = model.best_score['training'][hyperparams['metric']]
        score_val = model.best_score['valid_0'][hyperparams['metric']]
        fold_scores.append({'train':score_tr,'val':score_val})

        val_preds = model.predict(data.data.loc[val_index])
        y_oof.loc[val_index] = val_preds

        models.append(model)

    score_oof = calc_score(y_train, y_oof, config['mode'])

    score_tr_mean = np.mean([i['train'] for i in fold_scores])
    score_val_mean = np.mean([i['val'] for i in fold_scores])

    all_scores = {
                    'oof':score_oof,
                    'tr_mean':score_tr_mean,
                    'val_mean':score_val_mean,
                    }

    return all_scores, models, y_oof

@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:

    def save_preds(preds,config):
        result = pd.DataFrame({
            "line_id": config['line_id'],
            "prediction": preds,
        })

        result.to_csv(config['prediction_csv'], index=False)

    def predict_one(est, X): return est.predict(X)

    preds_mean = []
    score_type = 'score_oof'

    best_oof = np.min([i[score_type] for i in config["lgb_cv_models"]])
    scores = np.array([i[score_type] for i in config["lgb_cv_models"]])
    sorted_scores_ix = np.argsort((scores))
    models_for_stack = np.array(config["lgb_cv_models"])[sorted_scores_ix[:]]

    print('best oof :',best_oof)
    print(f'len(models_for_stack): {len(models_for_stack)}')

    for cv_model_data in models_for_stack:

        iteration_start = time.time()

        model_score = cv_model_data[score_type]
        score_difference = abs((model_score - best_oof) / best_oof)

        if score_difference > 0.03: 
            continue
            
        print('merging with oof: ', model_score)

        X_test_model_preds = []

        tasks = cv_model_data['models']
        func = predict_one
        func_params = {'X':X}
        with ThreadPool(processes=config["n_threads"]) as pool:
            results = list(pool.map(partial(func,**func_params),tasks))

        for preds in results:
            preds = pd.Series(preds)
            preds_mean.append(preds)
            X_test_model_preds.append(preds)

        test_preds_mean = pd.concat(X_test_model_preds,1).mean(1)

        iteration_time = time.time()-iteration_start
        elapsed_time = (time.time() - config['start_time']) 
        have_time = (config["time_limit"] - elapsed_time - iteration_time*2) > 35

        save_preds(test_preds_mean,config) #in case of running out of time

        if not have_time:
            break

    preds_mean = pd.concat(preds_mean,1).mean(1)

    if config["non_negative_target"]:
        preds_mean = [max(0, p) for p in preds_mean]

    return preds_mean

def save_model(models, hyperparams, scores, train_oof, stored_models_key, save_to_disk, config):
    config[stored_models_key].append(
        {
            'models':models,
            'hyperparams':hyperparams,
            'train_oof':train_oof,
            'score_oof':scores['oof'],
            'score_tr':scores['tr_mean'],
            'score_val':scores['val_mean'],
        }
    )
    if save_to_disk: config.save()


def calc_score(y_true, y_pred, mode):
    score = roc_auc_score(y_true, y_pred) if mode == "classification" else \
        np.sqrt(mean_squared_error(y_true, y_pred))
    return score
