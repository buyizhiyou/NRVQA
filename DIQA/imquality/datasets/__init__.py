import os

from .live_iqa import LiveIQA  # noqa
from .tid2013 import Tid2013  # noqa

DATASETS_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKSUMS_PATH = os.path.join(DATASETS_PATH, 'url_checksums')
CHECKSUMS_PATH = os.path.normpath(CHECKSUMS_PATH)
