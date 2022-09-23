import os

def ensure_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)