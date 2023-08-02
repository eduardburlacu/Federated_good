import os

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
os.chdir(PROJECT_PATH)

directories = os.listdir(PROJECT_PATH)
PATH={
    directory: os.path.join(PROJECT_PATH, directory) for directory in directories
}

src_directories = os.listdir(PATH['src'])
PATH_src = {
    directory: os.path.join(PATH['src'], directory) for directory in src_directories
}

path_sims = os.path.join(PATH['config'],'simulations')
sims  = os.listdir(path_sims)
PATH_sim = {
    os.path.splitext(f)[0]: os.path.join(path_sims, f) for f in sims
}

data_directories = os.listdir(os.path.join(PATH['FedProx'],'data'))

PATH_data = {
    os.path.splitext(f)[0]: os.path.join(path_sims, f) for f in data_directories
}
GOD_CLIENT_NAME = "952630398097868223647162069900715440297608885786503411514402181337302872670061123373871861"