#!/bin/bash
# Simple bash script to download and start the BERT service
# Details are here:
# https://github.com/hanxiao/bert-as-service#speech_balloon-faq

# Download the BERT model
# https://github.com/google-research/bert
# fname='uncased_L-12_H-768_A-12'  # BERT-Base, Uncased
fname='uncased_L-24_H-1024_A-16' # BERT-Large, Uncased

# Specify the number of workers. The server can handle up to num_worker
# concurrent requests.
num_worker = 1

cd /tmp/

if [ ! -f /tmp/${fname}.zip ]; then
    wget https://storage.googleapis.com/bert_models/2018_10_18/${fname}.zip /tmp
    unzip /tmp/${fname}.zip -d /tmp/
fi

# Start the BERT service
bert-serving-start -model_dir /tmp/${fname}/ -num_worker=${num_worker}&
