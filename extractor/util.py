import os
import yaml
import re
import logging.config
import signal
from contextlib import contextmanager

def load_log_config():
    env = os.getenv('ENV', 'prd')
    with open(os.path.join(os.path.dirname(__file__), f'../config/log_conf_{env}.yml'), 'r') as f:
        config = yaml.load(stream=f, Loader=yaml.SafeLoader)
    return config

def remove_space(text):
    if text is None:
        return ''
    return re.sub(r'\s', '', text)

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException()
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def clean_text(text):
    text = re.sub('\s*\n\s*', '\n', text)
    text = re.sub('\t|[ \t]{2,}', ' ', text)
    return text.strip()
