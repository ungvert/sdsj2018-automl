import argparse
from lib.util import timeit, log, send_signal_to_our_processes
from lib.automl import AutoML
import pandas as pd
import os
import subprocess
from pathlib import Path
import json
import time
from signal import SIGKILL, SIGTERM
from multiprocessing import Process
import psutil
import re

os.environ['OMP_NUM_THREADS'] = '1'
BUFFER_BEFORE_SENDING_SIGTERM = 7  
DELAY_TO_SIGKILL = 5  


@timeit
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['classification', 'regression'])
    parser.add_argument('--model-dir')
    parser.add_argument('--train-csv')
    parser.add_argument('--test-csv')
    parser.add_argument('--prediction-csv')
    args = parser.parse_args()

    automl = AutoML(args.model_dir)

    time_limit = int(os.environ['TIME_LIMIT'])
    automl.config['time_limit'] = time_limit
    log(f'{args.model_dir} - time_limit: {time_limit}')

    def automl_train():
        automl.train(args.train_csv, args.mode)
        # automl.save()

        out_log = pd.DataFrame(automl.config['log'])
        out_log.to_csv(f'{args.model_dir}/log_train.csv')
        print(pd.DataFrame(automl.config['log']))
        print('='*20)        

    def automl_predict():
        automl.load()
        automl.predict(args.test_csv, args.prediction_csv)


    time_left_for_task = time_limit - DELAY_TO_SIGKILL - BUFFER_BEFORE_SENDING_SIGTERM
    print('time_left_for_task :', time_left_for_task)

    if args.train_csv is not None:
    
        target_proc = automl_train

    elif args.test_csv is not None:

        target_proc = automl_predict

    else:
        exit(1)

    p = Process(target=target_proc,
                kwargs={
                        })
    p.start()
    p.join(time_left_for_task)
    pid = p.pid
    if p.is_alive():
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()

    log("Starting Shutdown!")

    program_exp = re.compile(r"main\.py").search
    contacts = send_signal_to_our_processes(sig=SIGTERM, filter=program_exp)
    log("Sending SIG=%d to %s" % (SIGTERM, str(contacts)))

    time.sleep(DELAY_TO_SIGKILL)

    contacts = send_signal_to_our_processes(sig=SIGKILL, filter=program_exp)
    log("Sending SIG=%d to %s" % (SIGKILL, str(contacts)))                

if __name__ == '__main__':
    main()
