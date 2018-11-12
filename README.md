# 10th place solution to Sberbank Data Science Journey 2018: AutoML

[SDSJ AutoML](https://sdsj.sberbank.ai/ru/contest) — AutoML(automatic machine learning) competition aimed at development of machine learning systems for processing banking datasets: transactions, time-series as well as classic table data from real banking operations.
Processing is handled automatically by the system with models selection, architecture, hyper-parameters, etc.

# Solution description

Based on [@tyz910](https://github.com/tyz910/) 's public [kernel](https://github.com/tyz910/sdsj2018)

## Preprocessing:

All preprocessing procedures prallelized.

- If dataset's size bigger than 2Gb -> filtering columns with Boruta
- Drop constant columns
- Add is_na columns
- Filling na values and downcasting
- Extracting features from datetime columns
- Target encoding for `string` columns

## Training:

While we have time:

1. Sample LightGBM hyperparameters and K-fold parameters(`n_splits`, `shuffle`)
2. Construct folds as subset from main train LightGBM dataset
3. Train all folds with LightGBM
4. Minimize oof-score with HyperOpt
4. Save all models and train another CV cycle if have time


## Local Validation

Public datasets for local validation: [sdsj2018_automl_check_datasets.zip](https://s3.eu-central-1.amazonaws.com/sdsj2018-automl/public/sdsj2018_automl_check_datasets.zip)


- Official [how to](https://github.com/sberbank-ai/sdsj2018-automl/blob/master/README_EN.md#how-to-local-validation)
- Baseline [example](https://github.com/sberbank-ai/sdsj2018-automl)
- [@vlarine](https://github.com/vlarine)'s public [kernel](https://github.com/vlarine/sdsj2018_lightgbm_baseline).

### Docker :whale:

`docker pull ungvert/sdsj2018`

