'''
Example usage
client1.py
if __name__=='__main__':
	client1 = Communicator('127.0.0.1',50123)
	client1.listen()
	msg=['Wello Horlf',7474]
	client1.send_msg(client1.connection,msg)

client2.py
	client2 = Communicator('127.0.0.1',30194)
	client2.connect('127.0.0.1',50123)
	print(client2.recv_msg(client2.sock))

'''

from signal import signal, SIGPIPE, SIG_DFL

signal(SIGPIPE, SIG_DFL)
import math
import time
import pickle
import struct
import socket
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Communicator(object):
	def __init__(self, ip_address, index):
		self.ip = ip_address
		self.index = index
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.connected = False
		self.connection = None

	def get_speed(self, model_size: float):
		if self.connected:
			network_time_start = time.time()
			msg = ['MSG_TEST_NETWORK', self.net.cpu().state_dict()]
			self.send_msg(
				self.sock,
				msg
			)
			msg = self.recv_msg(self.sock, 'MSG_TEST_NETWORK')[1]
			network_time_end = time.time()
			self.mbps = (2 * model_size * 8) / (network_time_end - network_time_start)  # Mbit/s
		else:  # Use -1 to mark lack of connection
			self.mbps = -1.

	def connect(self, other_addr, other_port):
		try:
			# logger.info(f'Connecting to {other_addr}:{other_port}.')
			self.sock.connect((other_addr, other_port))
			self.connected = True
			print(f"Connected to {other_addr}:{other_port}")
		except socket.error as e:
			logger.error(e)

	def listen(self):
		self.sock.bind((self.ip, self.index))
		self.sock.listen(1)
		print(f"Listening for connections on {self.ip}:{self.index}")

		while not self.connected:
			self.connection, address = self.sock.accept()
			print(f"Accepted connection from {address}")
			self.connected = True

	def disconnect(self, other_sock: socket.SocketType, init=True):
		if self.connected:
			if init:
				self.send_msg(other_sock,
							  f'Finished the transfer {str(self.sock.getpeername()[0])}:{str(self.sock.getpeername()[1])} <---> {str(other_sock.getpeername()[0])}:{str(other_sock.getpeername()[1])} \n Closing the connection...')
			self.sock.close()
			self.connected = False

	def send_msg(self, sock, msg):
		msg_pickle = pickle.dumps(msg)
		sock.sendall(struct.pack(">I", len(msg_pickle)))
		sock.sendall(msg_pickle)
		logger.debug(msg[0] + 'sent to' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

	def recv_msg(self, sock, expect_msg_type=None):
		enc = sock.recv(4)
		msg_len = struct.unpack(">I", sock.recv(4))[0]
		msg = sock.recv(msg_len, socket.MSG_WAITALL)
		msg = pickle.loads(msg)
		logger.debug(msg[0] + 'received from' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

		if expect_msg_type is not None:
			if msg[0] == 'Finish':
				return msg
			elif msg[0] != expect_msg_type:
				raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
		return msg

	def start(self):
		listen_thread = threading.Thread(target=self.listen)
		listen_thread.start()


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

	def start(self):
		listen_thread = threading.Thread(target=self.listen)
		listen_thread.start()

	def send_data(self, data):
		for connection in self.connections:
			try:
				connection.sendall(data.encode())
			except socket.error as e:
				print(f"Failed to send data. Error: {e}")

