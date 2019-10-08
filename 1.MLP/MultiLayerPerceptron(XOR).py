import tensorflow as tf

x_data = [[0, 0], [0, 1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0))
Wh = tf.Variable(tf.random_uniform([3, 1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([3], -1.0, 1.0))
bh = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

logits = tf.add(tf.matmul(X, W), b)
output = tf.nn.sigmoid(logits)

logitsh = tf.add(tf.matmul(output, Wh), bh)
outputh = tf.nn.sigmoid(logitsh)

cost = tf.reduce_mean(tf.square(outputh - Y))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(8000):
        for x, y in zip(x_data, y_data):
            _, cost_val = sess.run([train_op, cost], feed_dict={X:[x], Y:[y]})
        print(step, cost_val, sess.run(W), sess.run(b))
        print(step, cost_val, sess.run(Wh), sess.run(bh))

    print(sess.run(outputh, feed_dict={X:x_data}))