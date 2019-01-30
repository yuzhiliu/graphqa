import numpy as np
import tensorflow as tf
import pickle

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

pickle.dump(questions, open("questions_saved.p", "wb"))
pickle.dump(doc_vecs, open("doc_vecs_saved.p", "wb"))
"""
topk = 5
while True:
    query = input('your question: ')
    query_vec = bc.encode([query])[0]
    # compute simple dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], questions[idx]))
"""
# https://github.com/hanxiao/bert-as-service/blob/master/client/README.md
doc_label = [0] * len(questions) # a dummy list of all-zero labels

questions = 0
doc_vecs = 0
questions = pickle.load(open("questions_saved.p", "rb"))
doc_vecs = pickle.load(open("doc_vecs_saved.p", "rb"))

#fn_encoded = fname + ".encoded"
## write to tfrecord
#with tf.python_io.TFRecordWriter(fn_encoded) as writer:
#    def create_float_feature(values):
#        return tf.train.Feature(float_list=tf.train.FloatList(value=values))
#
#    def create_int_feature(values):
#        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
#
#    for (vec, label) in zip(doc_vecs, doc_label):
#        features = {'features': create_float_feature(vec), 'labels': create_int_feature([label])}
#        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
#        writer.write(tf_example.SerializeToString())
#
#def _decode_record(record):
#    """Decodes a record to a TensorFlow example."""
#    return tf.parse_single_example(record, {
#        'features': tf.FixedLenFeature([768], tf.float32),
#        'labels': tf.FixedLenFeature([], tf.int64),
#    })
#
#
#
#ds = (tf.data.TFRecordDataset(fn_encoded).repeat().shuffle(buffer_size=100).apply(
#    #tf.contrib.data.map_and_batch(lambda record: _decode_record(record), batch_size=64))
#    #  .make_one_shot_iterator().get_next())
#    tf.data.experimental.map_and_batch(lambda record: _decode_record(record), batch_size=64))
#      .make_one_shot_iterator().get_next())
#print(doc_vecs)
#print(ds)

topk = 10
while True:
    query = input('your question: ')
    query_vec = bc.encode([query])[0]
    # compute simple dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1)
    score.sort()
    print(score[::-1])
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], questions[idx]))
