import os

TIMEOUT = 12
DEFAULT_SERVER_ADDRESS = '127.0.0.1' #For socket bind
DEFAULT_GRPC_ADDRESS = "[::]:8080"
PORT_ROOT = 20000
MONITOR_PATH= os.path.join(os.path.abspath(os.path.dirname(__file__)), "Monitor")
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(PROJECT_PATH)

directories = os.listdir(PROJECT_PATH)
PATH={
    directory: os.path.join(PROJECT_PATH, directory) for directory in directories
}

src_directories = os.listdir(PATH['src'])
PATH_src = {
    directory: os.path.join(PATH['src'], directory) for directory in src_directories
}

path_data =os.path.join(PATH_src['Dataset'],'data')
data_directories = os.listdir(path_data)

PATH_data = {
    os.path.splitext(f)[0]: os.path.join(path_data, f) for f in data_directories
}
GOD_CLIENT_NAME = "952630398097868223647162069900715440297608885786503411514402181337302872670061123373871861"
SEED = 0
