import os

from platformdirs import user_cache_dir

CACHE_FILE = os.path.join(user_cache_dir("refcm"), "cache.json")
TREE_FILE = os.path.join(user_cache_dir("refcm"), "typetree.json")
LOG_FILE = os.path.join(user_cache_dir("refcm"), "log.log")
