import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src import PATH, PATH_src
from src.script.make_data_utils import make_data

def extract_mnist_data(is_embedded=False):
    mnist_in_dir  = os.path.join(PATH['FedProx'],'data', "mnist")
    mnist_out_dir = os.path.join(PATH_src['Dataset'],'data', 'mnist')
    make_data(mnist_in_dir, mnist_out_dir, is_embedded)
    os.chdir(os.path.dirname(__file__))

if __name__ == "__main__":
    extract_mnist_data()