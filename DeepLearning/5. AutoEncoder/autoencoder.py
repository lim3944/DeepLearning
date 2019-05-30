#2015004957 임현택 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 784])


W1 = tf.Variable(tf.random_uniform([784, 256],-1.0,1.0))
b1 = tf.Variable(tf.random_uniform([256],-1.0,1.0))

W2 = tf.Variable(tf.random_uniform([256,64],-1.0,1.))
b2 = tf.Variable(tf.random_uniform([64],-1.0,1.0))

W3 = tf.Variable(tf.random_uniform([64,256],-1.0,1.0))
b3 = tf.Variable(tf.random_uniform([256],-1.0,1.0))

W4 = tf.Variable(tf.random_uniform([256,784],-1.0,1.))
b4 = tf.Variable(tf.random_uniform([784],-1.0,1.0))

L1 = tf.nn.sigmoid(tf.matmul(X,W1)+b1)
L2 = tf.nn.sigmoid(tf.matmul(L1,W2)+b2)
L3 = tf.nn.sigmoid(tf.matmul(L2,W3)+b3)
decoder = tf.nn.sigmoid(tf.matmul(L3, W4)+b4)

cost = tf.reduce_mean(tf.square(X - decoder))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(300):
	total_cost = 0

	for i in range(total_batch):
		batch_xs,_ = mnist.train.next_batch(batch_size)
		batch_xs_noisy = [d + np.random.rand(784) for d in batch_xs]

		_, cost_val = sess.run([optimizer, cost], feed_dict = {X : batch_xs_noisy, Y : batch_xs})
		total_cost += cost_val

	print("Epoch : ", '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(total_cost/total_batch))



batch_xs,_ = mnist.train.next_batch(100)
batch_xs_noisy = [d + np.random.rand(784) for d in batch_xs]
samples = sess.run(decoder, feed_dict={X: batch_xs_noisy})
fig, ax = plt.subplots(3, 100, figsize=(sample_size, 2))
for i in range(100):
	ax[0][i].set_axis_off()
	ax[1][i].set_axis_off()
	ax[2][i].set_axis_off()
	ax[0][i].imshow(np.reshape(batch_xs[i], (28, 28)))
	ax[1][i].imshow(np.reshape(batch_xs_noisy[i],(28,28)))
	ax[2][i].imshow(np.reshape(samples[i], (28, 28)))
plt.show()

