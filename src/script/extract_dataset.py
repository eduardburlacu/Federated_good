import os
from src import PATH, PATH_src
from src.script.parse_config import get_variables
from src.script.make_data_utils import make_data

def extract_data(is_embedded=False):
    CONFIG = get_variables()
    dataset_name = CONFIG['DATASET'].name
    IN_PATH = os.path.abspath(
        os.path.join(PATH['leaf'],'data',dataset_name,'all_data')
    )
    OUT_PATH = os.path.join(PATH_src['Dataset'],'data', dataset_name)
    make_data(IN_PATH, OUT_PATH, is_embedded)
