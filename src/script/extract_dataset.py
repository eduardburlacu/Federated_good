import os
from src import PATH, PATH_src
from src.script.get_variables import get_variables

variables = get_variables()
dataset_name = get_variables()['DATASET'].name
IN_PATH = os.path.abspath(
    os.path.join(PATH['leaf'],'data',dataset_name,'all_data')
)
OUT_PATH = os.path.join(PATH_src['Dataset'],'data', dataset_name)
