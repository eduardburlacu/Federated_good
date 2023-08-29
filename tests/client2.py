from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

import time
import pickle
import struct
import socket
import threading
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Communicator(object):
	def __init__(self, ip_address, index ):
		self.ip = ip_address
		self.index = index
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.connected = False

	def connect(self,other_addr,other_port):
		while not self.connected:
			try:
				#logger.info(f'Connecting to {other_addr}:{other_port}.')
				self.sock.connect((other_addr, other_port))
				self.connected = True
				print(f"Connected to {other_addr}:{other_port}")
			except: continue


	def listen(self):
		self.sock.bind((self.ip, self.index))
		self.sock.listen(1)
		print(f"Listening for connections on {self.ip}:{self.index}")

		while True:
			connection, address = self.sock.accept()
			print(f"Accepted connection from {address}")
			self.connected = True

	def disconnect(self, sock:socket.SocketType, init=True):
		if self.connected and init:
			msg =[f'Finished the transfer \n Closing the connection...']
			self.send_msg(sock, msg)
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
	client2 = Communicator('127.0.0.1',30194)
	client2.connect('127.0.0.1', 50123)
	print(client2.recv_msg(client2.sock))
	client2.send_msg(client2.sock,['MSG_TEST_NETWORK','GOTCHA'])


	print(client2.recv_msg(client2.sock))
	msg= ['TEST PASSED',8532732957823]
	client2.send_msg(client2.sock,msg)
	client2.disconnect(client2.sock)
	print(client2.sock)

