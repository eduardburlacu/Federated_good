import time
from signal import signal, SIGPIPE, SIG_DFL
import sys
import pickle
import struct
import socket
import logging

signal(SIGPIPE,SIG_DFL)
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Communicator(object):

	def __init__(self, ip_address:str, index:int):
		self.ip = ip_address
		self.index = index
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.connected = False
		self.connection = None
		self.mbps = 0.


	def connect(self,other_addr,other_port):
		while not self.connected:
			try:
				#logger.info(f'Connecting to {other_addr}:{other_port}.')
				self.sock.connect((other_addr, other_port))
				self.connected = True
				print(f"Connected to {other_addr}:{other_port}")
			except:
				continue

	def listen(self, n:int=1):
		try:
			self.sock.bind((self.ip, self.index))
			self.sock.listen(n)
			print(f"Listening for connections on {self.ip}:{self.index}")

			while not self.connected:
				self.connection, address = self.sock.accept()
				print(f"Accepted connection from {address}")
				self.connected =True
		except:
			raise RuntimeError(f'Failed server setup with host number{self.ip}:{self.index}')
	def disconnect(self, other_sock:socket.SocketType, init=False):
		if self.connected:
			if init:
				self.send_msg(other_sock, f'Finished the transfer {str(other_sock.getpeername()[0])}:{str(other_sock.getpeername()[1])} \n Closing the connection...')
			self.sock.close()
			self.connected = False

	def to_socket(self)-> socket.SocketType:
		return self.connection if self.connection else self.sock

	def send_msg(self, sock, msg):
		dt = time.time()
		msg_pickle = pickle.dumps(msg)
		sock.sendall(struct.pack(">I", len(msg_pickle)))
		sock.sendall(msg_pickle)
		dt = time.time() - dt
		self.mbps = 1
		#self.mbps = (1.- Communicator.beta) * self.mbps + Communicator.beta * sys.getsizeof(msg_pickle)/ (dt * 1E6)
		logger.debug(msg[0]+'sent to'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

	def recv_msg(self, sock, expect_msg_type=None):
		msg_len = struct.unpack(">I", sock.recv(4))[0]
		msg = sock.recv(msg_len, socket.MSG_WAITALL)
		msg = pickle.loads(msg)
		logger.debug(msg[0]+'received from'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

		if expect_msg_type is not None:
			if msg[0] == 'Finish':
				return msg
			elif msg[0] != expect_msg_type:
				raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
		return msg

	def start(self):
		import threading
		listen_thread = threading.Thread(target=self.listen)
		listen_thread.start()