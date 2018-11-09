import os
import time
import pickle
from typing import Any
import psutil
import re
import pwd

nesting_level = 0
is_start = None

def timeit(method):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        config = None
        if'config' in kw: 
            config = kw['config']
        elif len(args)>1: 
            assert isinstance(args[-1],Config), 'No config passed in'
            config = args[-1] 

        if not is_start:
            print()

        is_start = True
        log("Start {}.".format(method.__name__))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        time_passed = end_time - start_time
        log("End {}. Time: {:0.2f} sec.".format(method.__name__, time_passed))
        if config is not None: 
            config['log'].append({"method":method.__name__,'time':time_passed}) 
        is_start = False
        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = "." * (4 * nesting_level)
    print("{}{}".format(space, entry))


class Config:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tmp_dir = model_dir
        self.data = {
            "start_time": time.time(),
            "verbose_eval":None,
            "log":[],
            "use_leak":False,

            "n_threads": 4,
            'seed':42,

            #prepocessing
            'numeric_categorical_threshold':30,
            'default_na_num_value':-1,
           
            #model related
            'train_num_boost_round': 100,
        }

    def is_train(self) -> bool:
        return self["task"] == "train"

    def is_predict(self) -> bool:
        return self["task"] == "predict"

    def is_regression(self) -> bool:
        return self["mode"] == "regression"

    def is_classification(self) -> bool:
        return self["mode"] == "classification"

    def time_left(self):
        return self["time_limit"] - (time.time() - self["start_time"])

    def save(self):
        with open(os.path.join(self.model_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(os.path.join(self.model_dir, "config.pkl"), "rb") as f:
            data = pickle.load(f)

        self.data = {**data, **self.data}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)


def rawcount(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines

def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum( buf.count(b'\n') for buf in f_gen )

def filter_pick(str_list, filter):
    match = list()
    try:
        match = [l for l in str_list for m in (filter(l),) if m]
    except IndexError:
        return False
    if len(match) == 1:
        return True
    else:
        return False

def send_signal_to_our_processes(filter, sig=0):
    # Sends sig to all processes matching filter
    contacts = list()
    processes = psutil.process_iter()
    this_process = psutil.Process()
    for proc in processes:
        try:
            if proc.pid == this_process.pid:
                continue
            if proc.username() == pwd.getpwuid(os.getuid()).pw_name:
                # = It is our process
                if filter_pick(str_list=proc.cmdline(), filter=filter):
                    proc.send_signal(sig)
                    contacts.append(proc.pid)
        except Exception as e:
            print(e)
    return contacts
