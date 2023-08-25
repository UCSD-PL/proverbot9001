#!/usr/bin/env python

from socketserver import StreamRequestHandler, TCPServer
import argparse

import json

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist

from rl import model_setup

model: nn.Module
optimizer: optim.Optimizer

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("-e", "--encoding-size", type=int, required=True)
  parser.add_argument("-p", "--port", default=9000, type=int)
  parser.add_argument("-l", "--learning-rate", default=5e-6, type=float)
  args = parser.parse_args()
  initializeState(args)

  with TCPServer(("0.0.0.0", args.port), ParamServerHandler) as server:
    server.serve_forever()

def initializeState(args: argparse.Namespace) -> None:
  global model
  global optimizer
  model = model_setup(args.encoding_size)
  optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)

class ParamServerHandler(StreamRequestHandler):
  def handle(self) -> None:
    global model
    global optimizer
    self.data = self.rfile.readline().decode('utf-8').strip()
    print(f"Received data: {self.data}")
    if self.data.startswith("GRAD:"):
      handle_grad_update(model, optimizer, json.loads(self.data[5:]))
    elif self.data == "GET PARAMS":
      print(json.dumps(model.state_dict()), file=self.wfile)
    else:
      print("Didn't recognize request!")

def handle_grad_update(model: nn.Module, optimizer: optim.Optimizer, gradients_dict: dict) -> None:
  for k, g in gradients_dict.items():
    model.get_parameter(k).grad = g
  optimizer.step()

if __name__ == "__main__":
  main()
