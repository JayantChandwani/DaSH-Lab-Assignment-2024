import os
import kagglehub

import sys
sys.path.append('gemma_pytorch')

from gemma.config import GemmaConfig, get_model_config
from gemma.model import GemmaForCausalLM
from gemma.tokenizer import Tokenizer
import contextlib
import torch
import json
import datetime
import socket
import threading

torch.cuda.empty_cache()
batch_size = 16

HOST = "127.0.0.1"
PORT = 65432 

# Choose variant and machine type
VARIANT = '2b-it' #@param ['2b', '2b-it', '9b', '9b-it', '27b', '27b-it']
MACHINE_TYPE = 'cuda' #@param ['cuda', 'cpu']

CONFIG = VARIANT[:2]
if CONFIG == '2b':
  CONFIG = '2b-v2'

weights_dir = kagglehub.model_download(f'google/gemma-2/pyTorch/gemma-2-{VARIANT}')

# Ensure that the tokenizer is present
tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

# Ensure that the checkpoint is present
ckpt_path = os.path.join(weights_dir, f'model.ckpt')
assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'

# Set up model config.
model_config = get_model_config(CONFIG)
model_config.tokenizer = tokenizer_path
model_config.quant = 'quant' in VARIANT

# Instantiate the model and load the weights.
torch.set_default_dtype(model_config.get_dtype())
device = torch.device(MACHINE_TYPE)
model = GemmaForCausalLM(model_config)
model.load_weights(ckpt_path)
model = model.to(device).eval()

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{PROMPT}<end_of_turn><eos>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{PROMPT}<end_of_turn><eos>\n"

def get_response_data(questions):
  questions = questions.split('\n')
  response_data = []
  for question in questions:   
    TimeSent = int(datetime.datetime.timestamp(datetime.datetime.now()))
    prompt = (
        USER_CHAT_TEMPLATE.format(PROMPT=question)
        + "<start_of_turn>model\n"
    )
    answer = model.generate(
        USER_CHAT_TEMPLATE.format(PROMPT=prompt),
        device=device,
        output_len=32,
    )    
    TimeRecvd = int(datetime.datetime.timestamp(datetime.datetime.now()))
    response_data.append(
        {
            "Prompt": question,
            "Message": answer,
            "TimeSent": TimeSent,
            "TimeRecvd": TimeRecvd,
            "Source": "Gemma-2b-it"
        }
    )
  return response_data

def handle_client(client_socket, address):
  try:
    while True:
      questions = client_socket.recv(2048).decode('utf-8')
      if not questions:
        break

      response = get_response_data(questions)
      response_data = json.dumps(response)
      client_socket.send(response_data.encode('utf-8'))
  except ConnectionResetError:
    print(f"Connection lost")
  finally:
    client_socket.close()
    print(f"Connection closed")

def server_init():
  # socket.AF_INET -> IPv4, socket.SOCK_STREAM -> TCP
  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.bind((HOST, PORT))
  server.listen(5)
  print(f"Server listening on port {PORT}")

  while True:
    client_socket, addr = server.accept()
    threads = []
    client_thread = threading.Thread(target=handle_client, args=(client_socket, addr))
    # threads.append(client_thread)
    client_thread.start()
    # client_thread.join()


if __name__ == "__main__":
  server_init()
