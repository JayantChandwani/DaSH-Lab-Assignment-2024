import os
import kagglehub

kagglehub.login()

import sys
sys.path.append('gemma_pytorch')

from gemma.config import GemmaConfig, get_model_config
from gemma.model import GemmaForCausalLM
from gemma.tokenizer import Tokenizer
import contextlib
import torch
import json
import datetime

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


# Chat templates
USER_CHAT_TEMPLATE = "<start_of_turn>user\n{PROMPT}<end_of_turn><eos>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{PROMPT}<end_of_turn><eos>\n"

questions = []

with open("input.txt", "r") as input_file:
  questions = input_file.readlines()

output = []
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
    output.append(
        {
            "Prompt": question,
            "Message": answer,
            "TimeSent": TimeSent,
            "TimeRecvd": TimeRecvd,
            "Source": "Gemma-2b-it"
        }
    )

with open("output.json", "w") as file:
    file.write(json.dumps(output))

