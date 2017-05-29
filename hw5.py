#!/usr/bin/python
from __future__ import print_function
import numpy as np
import tensorflow as tf
import math as math
import argparse
myfilenm = "mtraining.dat"
#np.set_printoptions(threshold='nan')
#parser = argparse.ArgumentParser()
#parser.add_argument('dataset')
#args = parser.parse_args()

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def read_from_csv(filename_queue):
  reader = tf.TextLineReader(skip_header_lines=1)
  _, csv_row = reader.read(filename_queue)
  record_defaults = [[1],[1],[1],[1]]
  col1, col2, col3, col4 = tf.decode_csv(csv_row, record_defaults=record_defaults)
  features = tf.stack([col1,col2,col4])  
  label = tf.stack([col3])  
  return features, label

def input_pipeline(filenm, batch_size, epo):
  filename_queue = tf.train.string_input_producer(["mtraining.dat"], num_epochs=epo, shuffle=False)  
  example, label = read_from_csv(filename_queue)
  print (example)
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  #print (sess.run(example_batch))
  print("end fn inputpipeline")
  return example_batch, label_batch

file_length = file_len(myfilenm) - 1
#examples, labels = input_pipeline(myfilenm,file_length, 1)
examples, labels = input_pipeline(myfilenm,1000, 1)
print ("file_length",file_length)
with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  sess.run(tf.local_variables_initializer()) 
  #tf.global_variables_initializer().run()

  #tf.initialize_all_variables().run()

  # start populating filename queue
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  #print (sess.run(examples))
  print("before try")
  i = 0
  try:
    while not coord.should_stop():
      print("in while")
      example_batch, label_batch = sess.run([examples, labels])
      print(example_batch)
      i += 1
  except tf.errors.OutOfRangeError:
    print('Done training, epoch reached')
  finally:
    print ("in finally i:",i)
    coord.request_stop()

  coord.join(threads) 
