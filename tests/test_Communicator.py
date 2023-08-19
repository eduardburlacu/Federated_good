#----Insert main project directory so that we can resolve the src imports-------
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, src_path)
#----------------------------Internal Imports-----------------------------
from src.Delegated.Communication import *

def test_communicator():
    com = Communicator(0, '127.0.0.1')
    print(com.sock)
    assert com.index==0
    assert com.connected == False
