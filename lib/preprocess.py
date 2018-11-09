import copy
import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,KFold
from lib.features import select_features
from lib.util import timeit, log, Config

from multiprocessing.pool import ThreadPool,Pool
from functools import partial
import gc

@timeit
def preprocess(df: pd.DataFrame, config: Config):

    feature_selection(df, config)
    df = preprocess_pipeline(df, config)
    gc.collect()

    return df

def preprocess_pipeline(df: pd.DataFrame, config: Config):

    drop_columns(df, config)

    date_cols = list(df.filter(like='datetime_'))
    str_cols = list(df.filter(like='string_'))
    num_cols = list(df.filter(like='number_'))
    id_cols = list(df.filter(like='id_'))

    for c in id_cols+num_cols:
        if str(df[c].dtype) == 'object':
            log(f'column {c} is object (expected numerical type), casted as category')
            df[c] = df[c].astype('category').cat.as_ordered().cat.codes

    df = add_is_na_cols(df,config)
    df = fillna(df, config)
    df = downcast(df,config)

    non_negative_target_detect(df, config)

    if len(date_cols)!=0: 
        df = process_datetime(df, date_cols, config)

    if len(str_cols)!=0: 
        df = process_strings(df, str_cols, config)
        df = mean_encode_kf(df, str_cols, 5, config)

    return df

def encode_col(col:str, df: pd.DataFrame,target_col:str,target_mean_global,alpha,n_folds):

    nrows_cat = df.groupby(col)[target_col].count()
    target_means_cats = df.groupby(col)[target_col].mean()
    target_means_cats_adj = (target_means_cats*nrows_cat + 
                                target_mean_global*alpha)/(nrows_cat+alpha)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    encoded_folds = []
    for n_fold, (train_index, val_index) in enumerate(kf.split(df)):

        df_for_estimation, df_estimated = df.iloc[train_index], df.iloc[val_index]
        nrows_cat = df_for_estimation.groupby(col)[target_col].count()
        target_means_cats = df_for_estimation.groupby(col)[target_col].mean()
        target_means_cats_adj = (target_means_cats*nrows_cat + 
                                    target_mean_global*alpha)/(nrows_cat+alpha)
        encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
        encoded_folds.append(encoded_col_train_part)

    encoded_col_train = pd.concat(encoded_folds, axis=0)
    encoded_col_train.fillna(target_mean_global, inplace=True)

    # encoded_col_train.name += '_(mean_enc)'
    return encoded_col_train

def apply_enc(name_values_col,categorical_prior):
    c,values,col = name_values_col
    return col.apply(lambda x: values[x] if x in values else 
                                categorical_prior)

def mean_encode_kf(df: pd.DataFrame, cols_to_enc:list, n_folds:int, config: Config):
    
    if "categorical_columns" not in config:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

        target_col = 'target'
        alpha = 5
        
        encoded_cols = []
        target_mean_global = config["categorical_prior"] = df[target_col].mean()
        config["categorical_columns"] = {} 

        for col in cols_to_enc:
            nrows_cat = df.groupby(col)[target_col].count()
            target_means_cats = df.groupby(col)[target_col].mean()
            target_means_cats_adj = (target_means_cats*nrows_cat + 
                                        target_mean_global*alpha)/(nrows_cat+alpha)
            
            config["categorical_columns"][col] = target_means_cats_adj.to_dict()

        tasks = cols_to_enc
        func = encode_col
        func_params = {'df':df,'target_col':target_col,
        'target_mean_global':target_mean_global,'alpha':alpha,'n_folds':n_folds}
        with Pool(processes=config["n_threads"]) as pool:
            results = list(pool.map(partial(func,**func_params),tasks))
    
        df = pd.concat([df.drop(cols_to_enc,1),pd.concat(results,1)], axis=1)

    else:
        tasks = [(c,values,df[c]) for c,values in config["categorical_columns"].items()]
        func = apply_enc
        func_params = {'categorical_prior':config["categorical_prior"]}
        with Pool(processes=config["n_threads"]) as pool:
            results = list(pool.map(partial(func,**func_params),tasks))
    
        df = pd.concat([df.drop(cols_to_enc,1),pd.concat(results,1)], axis=1)

    return df

def na_col(col):
    new_c = col.isna()
    new_c.name +='_isna'
    return new_c

@timeit
def add_is_na_cols(df: pd.DataFrame, config: Config):

    if "nan_columns" not in config:

        na_cols = df.isna().any()
        na_cols = na_cols[na_cols==True].index
        if len(na_cols)==0: return df

        tasks = [df[c] for c in df[na_cols]]
        func = na_col
        func_params = {}
        with Pool(processes=config["n_threads"]) as pool:
            results = list(pool.map(partial(func,**func_params),tasks))

        df = pd.concat([df, pd.concat(results, 1)], 1)

        config["nan_columns"] =na_cols
        log(f'na_cols: {len(na_cols)}')

    else:
        for c in config["nan_columns"]:
            df[f'{c}_isna'] = df[c].isna()

    return df

def transform_to_categorical(col):
    return col.astype('category').cat.as_ordered()

@timeit
def process_strings(df: pd.DataFrame, str_cols: list, config: Config):

    if "string_categorical_columns" not in config:
        tasks = [df[c] for c in str_cols]
        func = transform_to_categorical
        func_params = {}
        with Pool(processes=config["n_threads"]) as pool:
            results = list(pool.map(partial(func,**func_params),tasks))

        df = pd.concat([df.drop(str_cols, 1), pd.concat(results, 1)], 1)

        config["string_categorical_columns"] = {}
        for c in str_cols:
            config["string_categorical_columns"][c] = df[c].cat.categories
            df[c]= df[c].cat.codes

    else:
        for c, categories in config["string_categorical_columns"].items():
            df[c] = pd.Series(pd.Categorical(df[c], categories=categories, ordered=True)).cat.codes
    return df

def downcast_col(col):
    dtype = str(col.dtype)
    if dtype.startswith('float'):
        all_int = col.apply(float.is_integer).all()
        if all_int:
            return pd.to_numeric(col, downcast='integer')
        else:
            return pd.to_numeric(col, downcast='float')
    else:
        return pd.to_numeric(col, downcast='integer')

@timeit
def downcast(df: pd.DataFrame, config: Config):

    num_cols = list(df.filter(like='number_'))

    tasks = [df[c] for c in num_cols]
    func = downcast_col
    func_params = {}
    with Pool(processes=config["n_threads"]) as pool:
        results = list(pool.map(partial(func,**func_params),tasks))

    df = pd.concat([df.drop(num_cols,1), pd.concat(results,1)],1)
    return df


@timeit
def drop_columns(df: pd.DataFrame, config: Config):
    df.drop([c for c in ["is_test", "line_id"] if c in df], axis=1, inplace=True)
    drop_constant_columns(df, config)

@timeit
def fillna(df: pd.DataFrame, config: Config):
    for c in [c for c in df if c.startswith("number_")]:
        df[c].fillna(config['default_na_num_value'], inplace=True)

    for c in [c for c in df if c.startswith("string_")]:
        df[c].fillna("", inplace=True)

    for c in [c for c in df if c.startswith("datetime_")]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

    return df

@timeit
def drop_constant_columns(df: pd.DataFrame, config: Config):
    if "constant_columns" not in config:
        config["constant_columns"] = [c for c in df if df[c].nunique(dropna=False)<2]
        log("Constant columns: " + ", ".join(config["constant_columns"]))

    if len(config["constant_columns"]) > 0:
        df.drop(config["constant_columns"], axis=1, inplace=True, errors='ignore')

def collect_datetime_parts(col,date_parts,time_parts):
    added_cols = []

    for part in date_parts:
        part = part.lower()
        part_col_name = col.name + "_" + part
        part_data = getattr(col.dt, part)
        if part_data.nunique()>2:
            
            new_col = part_data.astype(np.uint16 if part == "year" else np.uint8)
            new_col.name = part_col_name

            added_cols.append(new_col)

    if col.dt.time.nunique()>2:
        for part in time_parts:

            part = part.lower()
            part_col_name = col.name + "_" + part
            part_data = getattr(col.dt, part)
            if part_data.nunique()>2:
                new_col = part_data.astype(np.uint16 if part == "year" else np.uint8)
                new_col.name = part_col_name

                added_cols.append(new_col)

    return added_cols

@timeit
def process_datetime(df: pd.DataFrame, date_cols:list, config: Config):

    date_parts = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 
                'Is_year_end', 'Is_year_start']

    time_parts = ['Hour', 'Minute', 'Second']
    
    if "date_columns" not in config:
        config["date_columns"] = {}


        tasks = [df[c] for c in date_cols]
        func = collect_datetime_parts
        func_params = {'date_parts':date_parts,
                       'time_parts':time_parts,}
        with Pool(processes=config["n_threads"]) as pool:
            results = list(pool.map(partial(func,**func_params),tasks))

        new_cols = []
        for added_cols in results:
            if len(added_cols):
                for col in added_cols:
                    c_name = col.name.split('_')
                    part = c_name[-1]
                    c = col.name.replace(f'_{part}','')
                    if c not in config["date_columns"]: config["date_columns"][c] = []
                    config["date_columns"][c].append(part)
                    new_cols.append(col)
            
        df = pd.concat([df.drop(date_cols,1), pd.concat(new_cols,1)],1)
        
    else:
        for c, parts in config["date_columns"].items():
            for part in parts:
                part_col = c + "_" + part
                df[part_col] = getattr(df[c].dt, part)
            df.drop(c, axis=1, inplace=True)
    
    return df        

@timeit
def scale(df: pd.DataFrame, config: Config):
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    scale_columns = [c for c in df if c.startswith("number_") and df[c].dtype != np.int8 and
                     c not in config["categorical_columns"]]

    if len(scale_columns) > 0:
        if "scaler" not in config:
            config["scaler"] = StandardScaler(copy=False)
            config["scaler"].fit(df[scale_columns])

        df[scale_columns] = config["scaler"].transform(df[scale_columns])

@timeit
def non_negative_target_detect(df: pd.DataFrame, config: Config):
    if config.is_train():
        config["non_negative_target"] = df["target"].lt(0).sum() == 0

@timeit
def feature_selection(df: pd.DataFrame, config: Config):
    if config.is_train():
        df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if df_size_mb < 2 * 1024:
            return

        selected_columns = []
        for i in range(3):

            config_sample = copy.deepcopy(config)

            df_sample = df.sample(frac=0.05, random_state=i).copy(deep=True)
            df_sample = preprocess_pipeline(df_sample, config_sample)
            y = df_sample["target"]
            X = df_sample.drop("target", axis=1)

            if len(X.columns) > 0:
                selected_columns += select_features(X, y, config["mode"])
            else:
                break

            df_size_mb = df.drop(list(set(df)-set(selected_columns)),1, errors='ignore').memory_usage(deep=True).sum() / 1024 / 1024
            if df_size_mb < 2 * 1024:
                break

        selected_columns = list(set(selected_columns))

        log("Selected columns: {}".format(selected_columns))

        drop_number_columns = [c for c in df if (c.startswith("number_") or c.startswith("id_")) and c not in selected_columns]
        if len(drop_number_columns) > 0:
            config["drop_number_columns"] = drop_number_columns


        drop_datetime_columns = [c for c in df if c.startswith("datetime_") and c not in selected_columns]
        if len(drop_datetime_columns) > 0:
            config["drop_datetime_columns"] = drop_datetime_columns

    if "drop_number_columns" in config:
        log("Drop number columns: {}".format(config["drop_number_columns"]))
        df.drop(config["drop_number_columns"], axis=1, inplace=True)

    if "drop_datetime_columns" in config:
        log("Drop datetime columns: {}".format(config["drop_datetime_columns"]))
        df.drop(config["drop_datetime_columns"], axis=1, inplace=True)
