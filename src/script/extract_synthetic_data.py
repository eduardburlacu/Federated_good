import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src import PATH, PATH_src
from src.script.make_data_utils import make_data

def extract_synthetic_data(data_type, is_embedded=False):
    synthetic_in_dir = os.path.join(PATH['FedProx'], f"data_{data_type}")
    synthetic_out_dir = os.path.join(PATH_src['Dataset'],'data', f'synthetic_{data_type}')
    make_data(synthetic_in_dir, synthetic_out_dir)

if __name__ == "__main__":
    extract_synthetic_data(sys.argv[1])