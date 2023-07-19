import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src import PATH, PATH_src
from src.script.make_data_utils import make_data
def extract_femnist_data(is_embedded=False):
    IN_PATH = os.path.abspath(
        os.path.join(PATH['leaf'],'data','nist','all_data')
    )
    OUT_PATH = os.path.join(PATH_src['Dataset'],'data', 'femnist')
    make_data(IN_PATH, OUT_PATH, is_embedded)

if __name__ == "__main__":
    extract_femnist_data()