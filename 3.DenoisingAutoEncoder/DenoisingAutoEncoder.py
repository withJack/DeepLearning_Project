import tensorflow as tf
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 784])

#assignment

#dropout을 얼마나 시킬 것인지에 대한 placeholder 설정
keep_prob1 = tf.placeholder(tf.float32)
# keep_prob2 = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([784, 500], -1., 1.))
b1 = tf.Variable(tf.random_uniform([500], -1., 1.))
L1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob1)

W2 = tf.Variable(tf.random_uniform([500, 64], -1., 1.))
b2 = tf.Variable(tf.random_uniform([64], -1., 1.))
L2 = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob1)

#Output
Wo1 = tf.Variable(tf.random_uniform([64,500], -1., 1.))
bo1 = tf.Variable(tf.random_uniform([500], -1., 1.))
O1 = tf.nn.sigmoid(tf.matmul(L2, Wo1) + bo1)
O1 = tf.nn.dropout(O1, keep_prob1)

Wo = tf.Variable(tf.random_uniform([500, 784], -1., 1.))
bo = tf.Variable(tf.random_uniform([784], -1., 1.))
model = tf.nn.sigmoid(tf.matmul(O1, Wo) + bo)

cost = tf.reduce_mean(tf.square(model-Y))
# optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
optimizer = tf.train.AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam').minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(40):
    total_cost = 0
    tmp = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs_noisy = batch_xs + np.random.rand(batch_size, 784)
        # _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y:batch_ys})
        # 학습 할 때 dropout 얼마나 시킬 것인지 설정 ex) 0.75 -> 75% node를 활성화 시키겠다.
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs_noisy, Y: batch_xs, keep_prob1: 0.75})
        tmp = total_cost
        total_cost += cost_val

    print('Epoch: ', '%04d' % (epoch+1),
          'Avg. cost = ', '{:.3f}'.format(total_cost/total_batch))


# is_correct = tf.equal(tf.arg_max(model, 1), tf.math.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
# print('정확도', sess.run(accuracy, feed_dict={X: mnist.test.images, Y:mnist.test.images, keep_prob1: 1}))

sample_size = 10

samples = sess.run(model,
                   feed_dict={X: mnist.test.images[:sample_size], keep_prob1: 1})

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))
    # fig.savefig("result" + str(i) + ".png")
    # ax.savefig("results" + str(i) + ".png")

plt.show()

