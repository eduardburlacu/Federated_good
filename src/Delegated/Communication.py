import pickle
import struct
import socket

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Communicator(object):
	def __init__(self, index, ip_address):
		self.index = index
		self.ip = ip_address
		self.sock = socket.socket()
		self.connected = False

	def connect(self,other_addr,other_port):
		try:
			logger.info('Connecting to Server.')
			self.sock.connect((other_addr, other_port))
			self.connected = True
		except socket.error as e:
			logger.error(e)

	def disconnect(self, other_sock:socket.SocketType, init=True):
		if self.connected:
			if init:
				self.send_msg(other_sock, f'Finished the transfer {str(self.sock.getpeername()[0])}:{str(self.sock.getpeername()[1])} <---> {str(other_sock.getpeername()[0])}:{str(other_sock.getpeername()[1])} \n Closing the connection...')
			self.sock.close()
			self.connected = False
	def send_msg(self, sock, msg):
		msg_pickle = pickle.dumps(msg)
		sock.sendall(struct.pack(">I", len(msg_pickle)))
		sock.sendall(msg_pickle)
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


class Peer:
	def __init__(self, host, port):
		self.host = host
		self.port = port
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.connections = []

	def connect(self, peer_host, peer_port):
		try:
			connection = self.socket.connect((peer_host, peer_port))
			self.connections.append(connection)
			print(f"Connected to {peer_host}:{peer_port}")
		except socket.error as e:
			print(f"Failed to connect to {peer_host}:{peer_port}. Error: {e}")

	def listen(self):
		self.socket.bind((self.host, self.port))
		self.socket.listen(10)
		print(f"Listening for connections on {self.host}:{self.port}")

		while True:
			connection, address = self.socket.accept()
			self.connections.append(connection)
			print(f"Accepted connection from {address}")

	def send_data(self, data):
		for connection in self.connections:
			try:
				connection.sendall(data.encode())
			except socket.error as e:
				print(f"Failed to send data. Error: {e}")

