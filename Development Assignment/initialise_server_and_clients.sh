#!/bin/bash

python3 server.py & 

sleep 30 &

for i in $(seq 1 3); do
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
    python3 client.py $i &
done

wait