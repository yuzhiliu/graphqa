import numpy as np
import tensorflow as tf
import pickle
from numpy import dot
from fuzzywuzzy import process
from bert_serving.client import BertClient

dirname = "/home/ubuntu/graphqa/notes/"
fname = "all_questions.txt"
fname = "questions-small.txt"
questions_path = dirname + fname

with open(questions_path) as fp:
    questions = fp.read().splitlines()
    print('%d questions loaded, avg. len of %d' % (len(questions), np.mean([len(d.split()) for d in questions])))
bc = BertClient(port=5555, port_out=5556)
doc_vecs = bc.encode(questions)

def cos_sim(a, b):
	"""
        Takes 2 vectors a, b and returns the cosine similarity according
	to the definition of the dot product.
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)


topk = 10
while True:
    query = input('your question: ')
    query_vec = bc.encode([query])[0]
    # Compute simple dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1)
    # Compute cosine similarity as score
    score = [cos_sim(query_vec, doc_vec) for doc_vec in doc_vecs]
    print(score[::-1])
    query_english = process.extractOne(query, questions)[0]
    print(query_english)
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], questions[idx]))
