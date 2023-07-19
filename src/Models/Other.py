import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src.Models.ClassModel import Model
from src.script.parse_config import DatasetNode

#-----------To be implemented--------------
class ResNet18(Model):
    pass

class ResNet20(Model):
    pass

def load_models_datanodes():
    mapper = {
        (ResNet20, DatasetNode('')),
        (ResNet18, DatasetNode('')),
    }
    return mapper

def load_datanodes():
    mapper = {
        DatasetNode(''),
        DatasetNode('')
    }
    return mapper

def load_models():
    mapper = {
        ResNet18,
        ResNet20,
    }
    return mapper
