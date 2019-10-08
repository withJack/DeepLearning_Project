import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot = True)
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob1 = tf.placeholder(tf.float32)

# kernel(filter)
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

'''
Tensor("Conv2D:0', shape=(?, 28, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# W = tf.get_variable(name=name + "_W", shape=[filter_size, filter_size, fin, fout], initializer = tf.contrib.layers.xavier_initializer()))
# b = tf.get_variable(name=name + "_b", shape=[fout], initializer=tf.contrib.layers.xavier_initializer(0.0))
# C = tf.nn.conv2d(din, W, strides=[1, 1, 1, 1], padding='SAME')
# R = tf.nn.relu(tf.nn.bias_add(C,b))

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.reshape(L2, [-1, 64 * 7 * 7])
L2 = tf.nn.dropout(L2, keep_prob1)

W3 = tf.get_variable("W3", shape=[64 * 7 * 7, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b
# hypothesis = tf.nn.dropout(hypothesis, keep_prob=keep_prob1)

learning_rate = 0.001
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y:batch_ys})
        # 학습 할 때 dropout 얼마나 시킬 것인지 설정 ex) 0.75 -> 75% node를 활성화 시키겠다.
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob1: 0.75})
        total_cost += cost_val

    print('Epoch: ', '%04d' % (epoch+1),
          'Avg. cost = ', '{:.3f}'.format(total_cost/total_batch))

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.math.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도', sess.run(accuracy, feed_dict={X: mnist.test.images, Y:mnist.test.labels, keep_prob1: 1}))
