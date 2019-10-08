import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

#assignment

#dropout을 얼마나 시킬 것인지에 대한 placeholder 설정
keep_prob1 = tf.placeholder(tf.float32)
# keep_prob2 = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([784, 1000], -1., 1.))
b1 = tf.Variable(tf.random_uniform([1000], -1., 1.))
L1 = tf.sigmoid(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob1)

W2 = tf.Variable(tf.random_uniform([1000, 10], -1., 1.))
b2 = tf.Variable(tf.random_uniform([10], -1., 1.))
model = tf.matmul(L1, W2) + b2
model = tf.nn.dropout(model, keep_prob1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
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

for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y:batch_ys})
        # 학습 할 때 dropout 얼마나 시킬 것인지 설정 ex) 0.75 -> 75% node를 활성화 시키겠다.
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob1: 0.75})
        total_cost += cost_val

    print('Epoch: ', '%04d' % (epoch+1),
          'Avg. cost = ', '{:.3f}'.format(total_cost/total_batch))

is_correct = tf.equal(tf.arg_max(model, 1), tf.math.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도', sess.run(accuracy, feed_dict={X: mnist.test.images, Y:mnist.test.labels, keep_prob1: 1}))

