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
  features = tf.stack([col1,col2])  
  label = tf.stack([col3])  
  return features, label

def input_pipeline(filenm, batch_size, epo):
  filename_queue = tf.train.string_input_producer(["mtraining.dat"], num_epochs=epo, shuffle=False)  
  example, label = read_from_csv(filename_queue)
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
examples, labels = input_pipeline(myfilenm,5, 1)
print ("file_length",file_length)

#----------------------------start building model:-----------------------
f_num = 2
x = tf.placeholder(tf.float32, [None,f_num]) #f_num is feature number in each data
#W = tf.Variable(tf.zeros([f_num, 1]))
W = tf.Variable(tf.random_normal([f_num, 5],stddev=0.35))
b = tf.Variable(tf.zeros([5]))
yt = tf.matmul(x,W) + b
print("yt!!!!!!!!!!!!!!!",yt)
y =tf.to_float( tf.argmax(tf.nn.softmax(tf.matmul(x, W) + b),1),name='ToFloat')
#vec = tf.nn.softmax(tf.matmul(x, W) + b)
#yint = tf.argmax(vec,1)
#y = tf.to_float(yint, name='ToFloat')
#y = vec# + vec[1,0]*2 + vec[2,0]*3 + vec[3,0]*4 + vec[4,0]*5
#y = x * W + b
print("y!!!!!!!!!!!!!!!",y)
#y_ = tf.placeholder(tf.float32,[None,1])
y_ = tf.placeholder(tf.float32,[None,1])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
accuracy = tf.reduce_mean(tf.reduce_sum(tf.square(y_-y)))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(accuracy)



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
      print("in ", i, " ")
      batch_xs, batch_ys = sess.run([examples, labels])
      if i == 11000 :
        break;
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
      print("W:", sess.run(W),"b:", sess.run(b))
      i += 1
  except tf.errors.OutOfRangeError:
    print('Done training, epoch reached')
  finally:
    print ("in finally i:",i)
    #coord.request_stop()
    
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    print("batch_xs:\n",batch_xs)
    #accuracy = tf.reduce_mean(tf.square(y_-y))
    print(sess.run(accuracy, feed_dict={x: batch_xs,
                                      y_: batch_ys}))
    coord.request_stop()

  coord.join(threads) 
