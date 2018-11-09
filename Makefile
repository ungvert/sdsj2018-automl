IMAGE=ungvert/sdsj2018
DOWNLOAD_URL=https://s3.eu-central-1.amazonaws.com/sdsj2018-automl/public/sdsj2018_automl_check_datasets.zip

ifeq ($(OS), Windows_NT)
	DOCKER_BUILD=docker build -t ${IMAGE} .
else
	DOCKER_BUILD=docker build -t ${IMAGE} . && (docker ps -q -f status=exited | xargs docker rm) && (docker images -qf dangling=true | xargs docker rmi) && docker images
endif

ifeq ($(DATASET),)
	DATASET=1
endif

DATASET_MODE=_c
ifeq ($(DATASET), $(filter $(DATASET), 1 2 3))
	DATASET_MODE=_r
endif

DATASET_NAME=${DATASET}${DATASET_MODE}

ifeq ($(DATASET_MODE), _r)
	TRAIN_MODE=regression
else
	TRAIN_MODE=classification
endif

DATA_PATH = data

TRAIN_CSV=${DATA_PATH}/check_${DATASET_NAME}/train.csv
TEST_CSV=${DATA_PATH}/check_${DATASET_NAME}/test.csv

PREDICTIONS_CSV=predictions/check_${DATASET_NAME}.csv
MODEL_DIR=models/check_${DATASET_NAME}

DOCKER_RUN=docker run --rm -it -v "${CURDIR}":/app -w /app --memory 12g --memory-swap 12g --cpuset-cpus=0-3 -e TIME_LIMIT=300 ${IMAGE}

download:
	${DOCKER_RUN} /bin/bash -c "test -f ${TRAIN_CSV} || (cd data && curl ${DOWNLOAD_URL} > data.zip && unzip data.zip && rm data.zip)"

train:
	${DOCKER_RUN} python3 main.py --mode ${TRAIN_MODE} --train-csv ${TRAIN_CSV} --model-dir ${MODEL_DIR}

predict:
	${DOCKER_RUN} python3 main.py --test-csv ${TEST_CSV} --prediction-csv ${PREDICTIONS_CSV} --model-dir ${MODEL_DIR}

score:
	${DOCKER_RUN} python3 score.py

docker-build:
	${DOCKER_BUILD}

docker-push:
	docker push ${IMAGE}

run-bash:
	${DOCKER_RUN} /bin/bash

run-jupyter:
	docker run --rm -it -v "${CURDIR}":/app -w /app -p 8888:8888 ${IMAGE} jupyter notebook --ip=0.0.0.0 --no-browser --allow-root  --NotebookApp.token='' --NotebookApp.password=''

submission:
	${DOCKER_RUN} /bin/bash -c "sed -i.bak 's~{image}~${IMAGE}~g' metadata.json && zip -9 -r submissions/submission_`date '+%Y%m%d_%H%M%S'`.zip main.py lib/*.py metadata.json && mv metadata.json.bak metadata.json"
