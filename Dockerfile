FROM continuumio/anaconda3:5.2.0

# Common packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        vim \
        wget \
        curl \
        git \
        zip \
        unzip && \
    rm -rf /var/lib/apt/lists/*

# Tensorflow doesn't support python 3.7 yet. See https://github.com/tensorflow/tensorflow/issues/20517
# Fix to install tf 1.10:: Downgrade python 3.7->3.6.6 and downgrade Pandas 0.23.3->0.23.2
RUN conda install -y python=3.6.6 && \
    pip install pandas==0.23.2 && \
    # Another fix for TF 1.10 https://github.com/tensorflow/tensorflow/issues/21518
    pip install keras_applications==1.0.4 --no-deps && \
    pip install keras_preprocessing==1.0.2 --no-deps

RUN conda install -y -c conda-forge tensorflow 

RUN conda install -y torchvision-cpu -c pytorch
RUN conda install -y pytorch-nightly-cpu -c pytorch
RUN conda install -y -c conda-forge lightgbm xgboost catboost
RUN conda install -y -c saravji boruta 
RUN conda install -y -c jaikumarm hyperopt 
