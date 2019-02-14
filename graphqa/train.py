# This code was based on https://github.com/Octavian-ai/english2cypher
# Octavian.ai developed.

import tensorflow as tf
import os.path
import numpy as np
import traceback
import yaml
from collections import Counter
import logging

logger = logging.getLogger(__name__)

from .input import gen_input_fn, EOS
from .model import model_fn
from .args import get_args
from .util import *
from .build_data import *

def train(args):
    """
    Main function to train the data.
    """
    tf.logging.set_verbosity(tf.logging.DEBUG)

    estimator = tf.estimator.Estimator(
        model_fn,
        model_dir=args["model_dir"],
        warm_start_from=args["warm_start_dir"],
        params=args)

    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:gen_input_fn(args, "eval"))
    steps_per_cycle = int(args["max_steps"]/args["predict_freq"])

    def do_train(max_steps):
        """
        Perform the actural training.
        """
        # max_steps is a bit awkward, but hey, this is tensorflow
        train_spec = tf.estimator.TrainSpec(input_fn=lambda:gen_input_fn(args, "train"), max_steps=max_steps)
        tf.estimator.train_and_evaluate(
            estimator,
            train_spec,
            eval_spec
        )

    def do_predict(max_steps):
        """ Perform predictions."""
        print("-----------------------")
        print("Predictions")

        stats = Counter()

        def get_formatted_predictions():
            predictions = estimator.predict(input_fn=lambda:gen_input_fn(args, "predict"))

            for prediction in predictions:
                o = {}
                for k, v in prediction.items():
                    if k == "input":
                        o[k] = prediction_to_english(v)
                    elif k in ["guided", "target"]:
                        o[k] = [prediction_to_cypher(v)]
                    elif k == "beam":
                        o[k] = [prediction_to_cypher(i) for i in v]

                for i in o["beam"]:
                    if i == o["target"]:
                        stats["correct"] += 1
                    else:
                        stats["incorrect"] += 1

                logger.debug(o)
                yield o

            print(stats)

        with tf.gfile.GFile(os.path.join(args["output_dir"], f"predictions-{max_steps}.yaml"), 'w') as file:
            yaml.dump_all(
                get_formatted_predictions(),
                file,
                default_flow_style=False,
                width=999)

    for i in range(args["predict_freq"]):
        max_steps = steps_per_cycle * (i+1)

        if not args["skip_training"]:
            do_train(max_steps)

        try:
            do_predict(max_steps)

            if args["skip_training"]:
                break

        except Exception:
            traceback.print_exc()
            pass


if __name__ == "__main__":

    def extend(parser):
        parser.add_argument('--skip-training',
                action='store_true',
                help="Whether to skip the training part of not. If yes, then do not perform the training and only perform one prediction.")
        parser.add_argument('--tokenize-data',
                action='store_true',
                help="Whether to perform the tokenize data process or not.")

    args = get_args(extend)

    if args["tokenize_data"]:
        expand_unknowns_and_partition(args)

    train(args)
