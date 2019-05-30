import tensorflow as tf
import numpy as np

x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[1,0], [0,1],[0,1],[1,0]]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2,3],-1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([3,2],-1.0, 1.0))
b1 = tf.Variable(tf.random_uniform([3],-1.0,1.0))
b2 = tf.Variable(tf.random_uniform([2],-1.0,1.0))

L1=tf.sigmoid(tf.matmul(X,W1)+b1)

model = tf.matmul(L1, W2)+b2

output_softmax = tf.nn.softmax(model)
output_argmax = tf.argmax(model,1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

opt = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train_op = opt.minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(8000):
		for x,y in zip(x_data, y_data):
			_,total_cost = sess.run([train_op, cost], feed_dict = {X : x_data, Y : y_data})
			'''
			if step%5 == 0:
				print(step, total_cost)
			'''
	print("predict: ", sess.run(model, feed_dict={X:x_data}))
	print("predict with softmax: ", sess.run(output_softmax, feed_dict={X:x_data}))
	print("predict with argmax: ", sess.run(output_argmax, feed_dict={X:x_data}))
