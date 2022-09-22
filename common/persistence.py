import os
import pickle
import shutil
from lsm import LSM


class PersistentDict:
    def __init__(self, *args, **kwargs):
        path = args[0]
        self.lsm = LSM(path)
        self.lsm.open()

    def __setitem__(self, key, value):
        self.lsm[str.encode(key)] = pickle.dumps(value)

    def __getitem__(self, key):
        return pickle.loads(self.lsm[str.encode(key)])

    def __contains__(self, key):
        return str.encode(key) in self.lsm

    def __delitem__(self, key):
        del self.lsm[str.encode(key)]

    @staticmethod
    def copy(src_path, dest_path):
        shutil.copyfile(src_path, dest_path)

    @staticmethod
    def drop(path):
        os.remove(path)

    def close(self):
        self.lsm.close()
