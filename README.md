# GraphaQA
[![Build Status](https://travis-ci.org/yuzhiliu/graphqa.svg?branch=week2)](https://travis-ci.org/yuzhiliu/graphqa)

GraphQA is a question answering (QA) system built over knowledge graphs (KG).

This is based on [english2cypher](https://github.com/Octavian-ai/english2cypher) [David Mack](https://medium.com/@DavidMack) built.

The trained model can transform English questions into
[Cypher](https://neo4j.com/developer/cypher-query-language/) queries and then
query answers from the [Neo4j](https://neo4j.com/) graph database.

The following is a screenshot of the GraphQA system. Currently it does not have
a user friendly API yet.
<p align="center">
<img width="80%" src="img/gqa_readme_example.jpg" />
<br>
Figure 1. An illustration of the GraphQA system.
</p>

## Requirements

- Python 3.6
- Neo4j (for inference only)
- Docker 17.03 or above
- Nvidia GPU (training instance)


Currently the code is only tested under Python 3.6. The code will not work
properly for Python version less than 3.6. This is mainly due to the use of
[*f-String*](https://realpython.com/python-f-strings/) in the code. If the
default Python version is not 3.6, one can either create a [*Virtual
Environment*](https://docs.python-guide.org/dev/virtualenvs/) or use
[*Conda*](https://docs.anaconda.com/anaconda/user-guide/tasks/switch-environment/).

## Installation:

Once Python 3.6 is installed, you can download this source code by running
```shell
git clone https://github.com/yuzhiliu/graphqa/
```

### Installing from source

To install all the dependencies needed for running the code, one can first go
to the package directory
```shell
cd graphqa
```
and then run
```shell
make install
```

### Installing using Docker

Docker provides a virtual machine with everything set up to run the code. We
have provided a *Dockerfile* for you to create your own Docker image.

Once you have [installed Docker](https://docs.docker.com/engine/installation/) and downloaded our *Dockerfile*,  you can create the Docker image by running
```shell
cd graphqa/docker
docker build -t graphqa:Insight19A .
```
**Do not ignore the "." in the above command.**

After the Docker image is built, you can run
```shell
docker run -i -t graphqa:Insight19A /bin/bash
```

Note that the Docker image only contains the packages necessary for training
the model.

## Training – How to train a GraphQA system

Let's train our GraphQA model, translating from English to
[Cypher](https://neo4j.com/developer/cypher-query-language/)! 

We will use [CLEV graph](https://github.com/Octavian-ai/clevr-graph) dataset
[Octavian-ai](https://www.octavian.ai/) created. The full dataset can be found
[here](https://drive.google.com/open?id=1r2BS07_2lB25Vlo6a9HiafewTGENmI80).

Run the following command to download and preprocess the data
```shell
python3 -m graphqa.build_data \
    --input-dir=./data/ \
    --skip-extract=False \
    --gqa-path=./data/gqa.yaml \
    --eval-holdback=0.2 \
    --predict-holdback=0.1  \
    --vocab-size=120
```

This will download the data automatically to ./data/gqa.zip, unzip the file,
preprocess the data, and extrat the data for training, validation, and testing.

The whole dataset is about 300MB and it will take a while to process the data.

Run the following command to start the training:
```shell
python3 -m graphqa.train \
    --skip-training=False \
    --tokenize-data=False \
    --output-dir=./output \
    --model-dir=./output/model \
    --max-steps=300 \
    --predict-freq=3 \
    --batch-size=128 \
    --num-units=1024 \
    --num-layers=2 \
    --beam-width=10 \
    --max-len-cypher=180 \
    --learning-rate=0.001 \
    --dropout=0.2
```

The above command trains a two-layer model with 1024-dim hidden units and
embeddings for max-steps/predict_freq = 100 epochs. A dropout value of 0.2 is
used.

The model is saved in the ./output/model directory.

Alternatively, one can try
```bash
python3 -m graphqa.train --quick
```
This will use smaller batch size, one layer network, et al., and existing small
test input files to train the model. The result will not be good at all though.
It might be helpful to use this to debug the code.

## Inference – How to use the model to build a QA system

Once you have trained your model, you can use it to translate previously unseen
English question to Cypher query language. This process is called inference.
Since the ultimate goal is to get answers from the graph database, you will
need to have a graph database installed to run the inference. The simplest way
is to run a Docker image created by [Andrew
Jefferson](https://neo4j.com/staff/andrew-jefferson/) and then create a
database by running
```shell
cd scripts
bash start-neo4j-database.sh
```

After the database is created, you can also access the database from a regular
browser window by typing [http://localhost:7474](http://localhost:7474) and
signing in with Username: neo4j, Password:goodpasswd. At this stage, the
database is empty.

You can then run the following script to start the inference
```shell
python3 -m graphqa.predict
```
This will load the data into Neo4j database. Currently there is only command
line interface available.

## Some technical details about the model

We will not go into details of the [seq2seq](https://google.github.io/seq2seq/)
model here. Instead, let's look at some other technical details that are used
on top of seq2seq.

### Bahdanau Attention 

The *attention mechanism* was first introduced by by [Bahdanau et
al.](https://arxiv.org/abs/1409.0473). The basic idea is to let the decoder pay
*attention* to the source sentence. This allows the decoder also to carry part
of the source sentence information while encoding. With this approach the
information is spread throughout the sequence of annotations. By doing so, the
algorithm improves the long sentence translations.

<p align="center">
<img width="80%" src="img/attention.jpg" />
<br>
Figure 1. An illustration of the *attention mechanism* proposed by Bahdanau et al.
</p>

### Bidirectional RNN Encoder

Bidirectionality on the encoder side generally gives better performance. It
puts two independent RNN together. The input sequence is put into one RNN in
normal order and the other RNN in reverse order.

<p align="center">
<img width="80%" src="img/bidirectional.jpg" />
<br>
Figure 1. An illustration of the Bidirectional RNN Encoder.
</p>

### Beam Search Decoder

Using beam search decoding can improve the code performance comparing with the
greedy decoding. The idea of the beam search is to keep top N (beam width)
token candidates at each step in memory. The next step is then expanded from
these N tokens. 

<p align="center">
<img width="80%" src="img/beam.jpg" />
<br>
Figure 1. An illustration of the beam search with beam width 2.
</p>

### Dropout

Dropout is a very simple way to prevent neural networks from overfitting. The
main idea of dropout is to randomly drop a fraction of the units from the
neural network during training. This prevents units from correlating too much. 
A dropout value of 0.2 means that 80% of the units is kept.


## Results

The main result is that the model can now handle more flexible questions.

<p align="center">
<img width="80%" src="img/model_comparision_1.jpg" />
<br>
Figure 1. The new model can answer more flexible questions.
</p>

<p align="center">
<img width="80%" src="img/model_comparision_2.jpg" />
<br>
Figure 1. The new model can answer more flexible questions.
</p>
