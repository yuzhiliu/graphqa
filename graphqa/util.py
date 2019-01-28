from collections import Counter
import numpy as np
import string
import re
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"
SPACE = "<space>"
special_tokens = [UNK, SOS, EOS, SPACE]

UNK_ID = special_tokens.index(UNK)
SOS_ID = special_tokens.index(SOS)
EOS_ID = special_tokens.index(EOS)

CYPHER_PUNCTUATION = "()[]-=\"',.;:?"
ENGLISH_PUNCTUATION = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~'


def load_vocab(args):
    """
    Load the tokens from the file constructed by "build_vocab" function in
    build_data.py.
	:type args: in practice never needed to construct args; only using get_args from args.py to construct the input.
    """
    tokens = list()

    with tf.gfile.GFile(args["vocab_path"]) as file:
        for line in file.readlines():
            tokens.append(line.replace("\n", ""))
            if len(tokens) == args["vocab_size"]:
                return tokens

    return tokens


def expand_unknown_vocab(line, vocab):
    """
    Treat the words that are in the "line" but not in the "vocab" as unknows,
    and expand the characters in those words as individual words.
    For example, the word "Spoon" is in "line" but not in "vocab", it will be
    expanded as "<S> <p> <o> <o> <n>".
    """
    ts = set(line.split(' '))
    unknowns = ts
    unknowns -= set(vocab)
    unknowns -= set([''])

    for t in unknowns:
        spaced = ''.join([f"<{c}> " for c in t])
        line = line.replace(t, spaced)

    return line


def pretokenize_general(text):
    """ Remove 's*$' and replace ' ' by ' <space> '"""
    text = re.sub(r'\s*$', '', text)
    text = text.replace(" ", f" {SPACE} ")
    return text


def pretokenize_cypher(text):
    """
    First remove 's*$' and replace ' ' by ' <space> '
    Then add spaces before and after "()[]-=\"',.;:?"
    Also treat spaces as tokens.
    """
    # In Cypher we want to tokenize punctuation as brackets are important
    # therefore we treat spaces as a token as well
    # so we can later reconstruct them

    text = pretokenize_general(text)

    for p in CYPHER_PUNCTUATION:
        text = text.replace(p, f" {p} ")
        # text = text.replace("  ", " ")
    return text


def pretokenize_english(text):
    """
    First remove 's*$' and replace ' ' by ' <space> '
    Then add spaces before and after '!"#$%&()*+,-./:;=?@[\\]^_`{|}~'
    """
    text = pretokenize_general(text)
    #for p in ENGLISH_PUNCTUATION:
    #    text = text.replace(p, f" {p} ")
    table = string.maketrans({p: f" {p} " for p in ENGLISH_PUNCTUATION})
    text.translate(table)

    # From Keras Tokenizer
    # filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    # split = ' '

    # translate_map = str.maketrans(filters, split * len(filters))
    # text = text.translate(translate_map)

    return text


def detokenize_specials(s, join=''):
    """ Detokenize the spacial characters. """
    try:
        end = s.index(EOS)
        s = s[0:end]
    except ValueError:
        pass

    for i in [UNK, SOS, EOS]:
        s = s.replace(i, "")

    s = s.replace(SPACE, " ")

    for i in string.ascii_lowercase:
        s = s.replace("<"+i+">"+join, i)
        s = s.replace("<"+i.upper()+">"+join, i.upper())

    return s


def detokenize_cypher(text):
    """ Detokenize Cypher. """
    for p in CYPHER_PUNCTUATION:
        text = text.replace(f" {p} ", p)

    text = detokenize_specials(text)
    return text


def detokenize_english(text):
    """ Detokenize English. """
    for p in ENGLISH_PUNCTUATION:
        text = text.replace(f" {p} ", p)

    return detokenize_specials(text)


def mode_best_effort(l):
    """Mode of list. Will return single element even if multi-modal"""

    if len(l) == 0:
        raise ValueError("Cannot compute mode of empty")

    c = Counter(l)
    return c.most_common(1)[0][0]


def prediction_row_to_cypher(pred):
    options = [prediction_to_cypher(i) for i in pred["beam"]]
    return mode_best_effort(options)

def prediction_to_(p, detokenize_fn):
    decode_utf8 = np.vectorize(lambda v: v.decode("utf-8"))
    p = decode_utf8(p)
    s = ''.join(p)
    s = detokenize_fn(s)
    return s

def prediction_to_english(p):
    return prediction_to_(p, detokenize_english)

def prediction_to_cypher(p):
    return prediction_to_(p, detokenize_cypher)


