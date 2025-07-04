import errno
import os


def mkdir(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e


def files_exist(files):
    return all([os.path.exists(f) for f in files])


def save_log(path, save_str):
    if os.path.exists(path):
        f = open(path, "a")
    else:
        f = open(path, "w")
    f.write(save_str)
    f.write("\n")
    return None
