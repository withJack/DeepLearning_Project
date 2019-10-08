import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)

# One hot encoding
sentence = ("if you want to build a ship, don't drum up people together to " "collect wood and don't assign them tasks and work, but rather " "teach them to long for the endless immensity of the sea.")
idx2char = list(set(sentence))
char2idx = {c: i for i, c in enumerate(idx2char)}

# hyper parameters
dic_size = len(char2idx)
hidden_size = len(char2idx)
num_classes = len(char2idx)
sequence_length = 10
learning_rate = 0.1

dataX = []
dataY = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char2idx[c] for c in x_str]  # x str to index
    y = [char2idx[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX) # 170

# placeholders
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
x_onehot = tf.one_hot(X, num_classes)
cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_onehot, initial_state=initial_state, dtype=tf.float32)

# FC layer
x_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(x_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(500):

        results, l, _ = sess.run([outputs, loss, train], feed_dict={X: dataX, Y: dataY})

        result = sess.run(prediction, feed_dict={X: dataX})

        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            print(i, j, ''.join([idx2char[t] for t in index]), l)

    results = sess.run(outputs, feed_dict={X: dataX})

    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:
            print(''.join([idx2char[t] for t in index]), end='')
        else:
            print(idx2char[index[-1]], end='')


