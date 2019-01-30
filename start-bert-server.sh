#!/bin/bash
# Simple bash script to download and start the BERT service
# Details are here:
# https://github.com/hanxiao/bert-as-service#speech_balloon-faq

# download the BERT model
fname='uncased_L-12_H-768_A-12'
fname='uncased_L-24_H-1024_A-16'
cd /tmp/

if [ ! -f /tmp/${fname}.zip ]; then
    wget https://storage.googleapis.com/bert_models/2018_10_18/${fname}.zip /tmp
    unzip /tmp/${fname}.zip -d /tmp/
fi

# start the BERT service
bert-serving-start -model_dir /tmp/${fname}/ -num_worker=4&
