# This code was based on https://github.com/Octavian-ai/english2cypher
# Octavian.ai developed.

# Make TF be quiet
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import argparse
import tensorflow as tf
import logging
import yaml
import traceback
import random
from neo4j.exceptions import CypherSyntaxError
import zipfile
import urllib.request
import pathlib
from fuzzywuzzy import process
import itertools

logger = logging.getLogger(__name__)

from .model import model_fn
from .util import *
from .args import get_args
from .input import gen_input_fn
from db import *

def translate(args, question):
    """ Translate Englisth into Cypher. """
    estimator = tf.estimator.Estimator(
        model_fn,
        model_dir=args["model_dir"],
        params=args)

    predictions = estimator.predict(input_fn=lambda: gen_input_fn(args, None, question))

    for p in predictions:
        # Only expecting one given the single line of input
        return prediction_row_to_cypher(p)


def print_examples(args):
    """ Print example questions. """
    with open(args['graph_path']) as file:
        for qa in yaml.load_all(file):
            if qa is not None:
                print("Example stations from graph:")
                stations = [i["name"] for i in qa["graph"]["nodes"][:8]]
                names = ', '.join(stations)
                print("> " + names + "\n")
                print("Example lines from graph:")
                lines = [i["name"] for i in qa["graph"]["lines"][:8]]
                names = ', '.join(lines)
                print("> " + names + "\n")

    with tf.gfile.GFile(args["questions_path"], "w") as q_file:
        for s1, s2 in itertools.combinations(stations, 2):
            q_file.write(f"Are {s1} and {s2} on the same line?\n")
            q_file.write(f"Are {s2} and {s1} on the same line?\n")

        for s in stations:
            q_file.write(f"How clean is {s}?\n")
            q_file.write(f"How big is {s}?\n")
            q_file.write(f"What music plays at {s}?\n")
            q_file.write(f"What architectural style is {s}?\n")
            q_file.write(f"Does {s} have disabled access?\n")
            q_file.write(f"Does {s} have rail connections?\n")
            q_file.write(f"Which lines is {s} on?\n")
            q_file.write(f"How many lines is {s} on?\n")
            q_file.write(f" {s}?\n")

        for l in lines:
            q_file.write(f"How many architectural styles does {s} pass through?\n")
            q_file.write(f"How many music styles does {s} pass through?\n")
            q_file.write(f"How many size of station does {s} pass through?\n")
            q_file.write(f"How many stations playing classical does {s} pass through?\n")
            q_file.write(f"How many clean stations does {s} pass through?\n")
            q_file.write(f"How many large stations does {s} pass through?\n")
            q_file.write(f"How many stations with disabled access does {s} pass through?\n")
            q_file.write(f"How many stations with rail connections does {s} pass through?\n")
            q_file.write(f"Which stations does {s} pass through?\n")

    a_station = lambda: random.choice(stations)
    a_line = lambda: random.choice(lines)
    print(f"""Example questions:
> Are {a_station()} and {a_station()} on the same line?
> How clean is {a_station()}?
> How big is {a_station()}?
> What music plays at {a_station()}?
> What architectural style is {a_station()}?
> Does {a_station()} have disabled access?
> Does {a_station()} have rail connections?
> Which lines is {a_station()} on?
> How many lines is {a_station()} on?
> How many architectural styles does {a_line()} pass through?
> How many music styles does {a_line()} pass through?
> How many sizes of station does {a_line()} pass through?
> How many stations playing classical does {a_line()} pass through?
> How many clean stations does {a_line()} pass through?
> How many large stations does {a_line()} pass through?
> How many stations with disabled access does {a_line()} pass through?
> How many stations with rail connections does {a_line()} pass through?
> Which stations does {a_line()} pass through?
""")

    print(f"""Example questions:
> How clean is {a_station()}?
> How big is {a_station()}?
> What music plays at {a_station()}?
> What architectural style is {a_station()}?
> Does {a_station()} have disabled access?
> Does {a_station()} have rail connections?
> How many architectural styles does {a_line()} pass through?
> How many music styles does {a_line()} pass through?
> How many sizes of station does {a_line()} pass through?
> How many stations playing classical does {a_line()} pass through?
> How many clean stations does {a_line()} pass through?
> How many large stations does {a_line()} pass through?
> How many stations with disabled access does {a_line()} pass through?
> How many stations with rail connections does {a_line()} pass through?
> Which lines is {a_station()} on?
> How many lines is {a_station()} on?
> Are {a_station()} and {a_station()} on the same line?
> Which stations does {a_line()} pass through?
""")


def download_model(args):
    """ Download the model from GoogleCloud. """
    if not tf.gfile.Exists(os.path.join(args["model_dir"], "checkpoint")):
        zip_path = "./model_checkpoint.zip"
        print("Downloading model (850mb)")
        urllib.request.urlretrieve ("https://storage.googleapis.com/octavian-static/download/english2cypher/model_checkpoint.zip", zip_path)
        print("Downloading vocab for model")
        assert args["vocab_path"][0:len(args["input_dir"])] == args["input_dir"], "Vocab path must be inside input-dir for automatic download"
        pathlib.Path(args["input_dir"]).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve ("https://storage.googleapis.com/octavian-static/download/english2cypher/vocab.txt", args["vocab_path"])
        print("Unzipping")
        pathlib.Path(args["model_dir"]).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path,"r") as zip_ref:
            zip_ref.extractall(args["model_dir"])


def load_questions(args):
    """ Read in the predefined questions. """
    with open(args["questions_path"]) as fp:
        questions = fp.read().splitlines()
        print('%d questions loaded, avg. len of %d' % (len(questions), np.mean([len(d.split()) for d in questions])))
        return questions

def cos_sim(a, b):
    """
    Takes 2 vectors a, b and returns the cosine similarity according to the
    definition of the dot product.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def extract_one_bert(query, questions):
    """
    User BERT to find the most similar question from the questions. This
    can also be realized by fuzzywuzzy.
    BERT Server needs to be started first. See start-bert-server.sh.
    """
    bc = BertClient(port=5555, port_out=5556)
    query_vec = bc.encode([query])[0]
    doc_vecs = bc.encode(questions)
    score = [cos_sim(query_vec, doc_vec) for doc_vec in doc_vecs]
    idx = np.argsort(score)[::-1][0]
    return questions[idx]

if __name__ == "__main__":

    def add_args(parser):
        parser.add_argument("--graph-path",   type=str, default="./data/gqa-single.yaml")
        parser.add_argument("--neo-url",      type=str, default="bolt://localhost:7687")
        parser.add_argument("--neo-user",     type=str, default="neo4j")
        parser.add_argument("--neo-password", type=str, default="goodpasswd")
        parser.add_argument("--questions-path",   type=str, default="./data/all_questions2.txt")

    args = get_args(add_args)

    logging.basicConfig()
    logger.setLevel(args["log_level"])
    logging.getLogger('graphqa').setLevel(args["log_level"])

    tf.logging.set_verbosity(tf.logging.ERROR)

    print_examples(args)

    download_model(args)

    questions = load_questions(args)

    with Neo4jSession(args) as session:
        logger.debug("Empty database")
        nuke(session)

        logger.debug("Load database")
        load_yaml(session, args["graph_path"])

        while True:
            query_english = str(input("Ask a question: ")).strip()
            print(query_english)
            # Pick one that is closest to the question asked
            query_english = process.extractOne(query_english, questions)[0]
            # query_english = extract_one_bert(query_english, questions)
            print(query_english)

            logger.debug("Translating...")
            query_cypher = translate(args, query_english)
            print(query_cypher)

            logger.debug("Run query")
            try:
                result = run_query(session, query_cypher)
            except CypherSyntaxError:
                print("Drat, that translation failed to execute in Neo4j!")
                continue
                #traceback.print_exc()
            else:
                all_answers = []
                for i in result:
                    for j in i.values():
                        all_answers.append(str(j))

                print(all_answers)
                if len(all_answers) == 0:
                    print("Answer: There is always an answer but I wouldn't tell you now. Try to ask another one")
                else:
                    print(f"Translation into cypher: '{query_cypher}'")
                    print()
                    print("Answer: " + ', '.join(all_answers))
                print()
