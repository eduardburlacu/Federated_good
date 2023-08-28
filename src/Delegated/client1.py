from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)
import math
import time
import pickle
import struct
import socket
import threading
import logging
from models import Net
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

net = Net(10)
class Communicator(object):
	def __init__(self, ip_address, index ):
		self.ip = ip_address
		self.index = index
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.connected = False
		self.connection = None

	def get_speed(self):

		if self.connected:
			network_time_start = time.time()
			msg = ['MSG_TEST_NETWORK', net.cpu().state_dict()]
			self.send_msg(
				self.connection,
				msg
			)
			msg = self.recv_msg(self.connection)[1]
			network_time_end = time.time()
			return (network_time_end - network_time_start)

		else:  # Use -1 to mark lack of connection
			return -1.

	def connect(self,other_addr,other_port):
		try:
			#logger.info(f'Connecting to {other_addr}:{other_port}.')
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
			self.connected =True

	def disconnect(self, other_sock:socket.SocketType, init=True):
		if self.connected and init:
			if init:
				self.send_msg(other_sock, f'Finished the transfer {str(other_sock.getpeername()[0])}:{str(other_sock.getpeername()[1])} \n Closing the connection...')
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

	def start(self):
		listen_thread = threading.Thread(target=self.listen)
		listen_thread.start()


if __name__=='__main__':
	client1 = Communicator('127.0.0.1', 50123)
	client1.listen()
	print(client1.get_speed())

	msg=['Wello Horld',7474]
	client1.send_msg(client1.connection, msg)
	print(client1.recv_msg(client1.connection))
	client1.disconnect(client1.sock, init=False)

