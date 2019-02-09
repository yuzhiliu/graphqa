# GraphaQA
[![Build Status](https://travis-ci.org/yuzhiliu/graphqa.svg?branch=week2)](https://travis-ci.org/yuzhiliu/graphqa)

GraphQA is a question answering (QA) system built ove knowledge graphs (KG).

This is based on [english2cypher](https://github.com/Octavian-ai/english2cypher) [David Mack](https://medium.com/@DavidMack) built.


## Requirements

- Python 3.6
- Neo4j (for inference only)
- Docker 17.03 or above
- Nvidia GPU (training instance)


Currently the code is only tested under Python 3.6. The code will not work
properly for Python version less than 3.6. This is mainly due to the use of
[*f-String*](https://realpython.com/python-f-strings/) in the code. If the
default Python version is not 3.6, one can either create a (*Virtual
Environment*](https://docs.python-guide.org/dev/virtualenvs/) or use
[*Conda*](https://docs.anaconda.com/anaconda/user-guide/tasks/switch-environment/).

## Installation:
Once Python 3.6 is installed, you can download this source code by running
```shell
git clone https://github.com/yuzhiliu/graphqa/
```

## Training – How to train a GraphQA system


```shell
make install
```

# Build the dataset to be used by training

Run
```bash
python -m graphqa.build_data
```
This will download the data from
[GoogleCloud](https://storage.googleapis.com/octavian-static/download/english2cypher/gqa.zip),
tokenlize the data, and partition them into three different files for training,
evaluation, and prediction. The files will be saved under data/ diretory as
"data/gqa.yaml", and six "txt" files by default.

# Training the model

Run
```bash
python -m graphqa.train
```
This will take a while to complete.

Alternatively, one can try
```bash
python -m graphqa.train --quick
```
This will use smaller batch size, one layer network, et al. and existing small
test input files to train the model. The result will not be good at all though.
It might be helpful to use this to debug the code.

# Running the model to make predictions

You will need to have a graph database installed to make predictions. The
simplest way is to run a Docker image created by [Andrew
Jefferson](https://neo4j.com/staff/andrew-jefferson/) and then create a
database by 

```bash
bash start-neo4j-database.sh
```

One can then run the following script to make predictions
```bash
python -m graphqa.predict
```
