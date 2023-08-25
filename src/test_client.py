#!/usr/bin/env python

import socket
import sys
import argparse
import os

from time import sleep

PORT = 9000

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-H", "--hostname")
  parser.add_argument("-p", "--port", default=9000, type=int)
  args = parser.parse_args()
  workerid = int(os.environ['SLURM_ARRAY_TASK_ID'])

  message = f"GET PARAMS"

  while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.connect((args.hostname, args.port))
      sock.sendall(bytes(message + "\n", "utf-8"))
    
      received = str(sock.recv(1024), "utf-8")
    
    print(f"Sent: {message}", file=sys.stderr)
    print(f"Received: {received}", file=sys.stderr)
    sleep(2)

if __name__ == "__main__":
  main()
