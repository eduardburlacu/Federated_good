import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src import PATH, PATH_src
from src.script.make_data_utils import make_data

def extract_shakespeare_data(is_embedded=True):
    shakespeare_in_dir = os.path.join(PATH['leaf'], 'data', 'shakespeare', 'data', 'all_data')
    shakespeare_out_dir = os.path.join(PATH_src['Dataset'], 'data','shakespeare')
    make_data(shakespeare_in_dir, shakespeare_out_dir, is_embedded)

if __name__ == "__main__":
    extract_shakespeare_data()