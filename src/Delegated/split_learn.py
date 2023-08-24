'''
Taken from FedAdapt paper implementation
'''
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import pickle, struct, socket
import collections
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def send_msg(sock, msg):
	msg_pickle = pickle.dumps(msg)
	sock.sendall(struct.pack(">I", len(msg_pickle)))
	sock.sendall(msg_pickle)
	logger.debug(msg[0]+'sent to'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

def recv_msg(sock, expect_msg_type=None):
	msg_len = struct.unpack(">I", sock.recv(4))[0]
	msg = sock.recv(msg_len, socket.MSG_WAITALL)
	msg = pickle.loads(msg)
	logger.debug(msg[0]+'received from'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

	if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
		raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
	return msg

def split_model(model:nn.Module, n1:int)-> Tuple[nn.Module, nn.Module]:
	depth = list(model.children())
	if n1 > len(depth): raise ValueError("n1 exceeds the total number of depth in the model.")
	net1 = nn.Sequential(*depth[:n1])
	net2 = nn.Sequential(*depth[n1:])

	return net1, net2

def split_weights_client(weights,cweights):
	for key in cweights:
		assert cweights[key].size() == weights[key].size()
		cweights[key] = weights[key]
	return cweights

def split_weights_server(weights,cweights,sweights):
	ckeys = list(cweights)
	skeys = list(sweights)
	keys = list(weights)

	for i in range(len(skeys)):
		assert sweights[skeys[i]].size() == weights[keys[i + len(ckeys)]].size()
		sweights[skeys[i]] = weights[keys[i + len(ckeys)]]

	return sweights

def concat_weights(weights,cweights,sweights):
	concat_dict = collections.OrderedDict()

	ckeys = list(cweights)
	skeys = list(sweights)
	keys = list(weights)

	for i in range(len(ckeys)):
		concat_dict[keys[i]] = cweights[ckeys[i]]

	for i in range(len(skeys)):
		concat_dict[keys[i + len(ckeys)]] = sweights[skeys[i]]

	return concat_dict



