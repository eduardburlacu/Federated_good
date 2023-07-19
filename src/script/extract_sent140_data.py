import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src import PATH, PATH_src
from src.script.make_data_utils import make_data

def extract_sent140_data():
    sent140_in_dir = os.path.join(PATH['leaf'], 'data', 'sent140', 'data', 'all_data')
    sent140_out_dir = os.path.join(PATH_src['Dataset'],'data', 'sent140')
    make_data(sent140_in_dir, sent140_out_dir)

if __name__ == "__main__":
    extract_sent140_data()
