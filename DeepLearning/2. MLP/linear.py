import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

hypothesis = W+X+b

cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(3000): #efork
		_, cost_val = sess.run([train_op, cost], feed_dict = {X : x_data, Y : y_data})#_는 불필요한값을 쓸때 
		print(step, cost_val, sess.run(W), sess.run(b))

	print("X : 6, Y:", sess.run(hypothesis, feed_dict={X:6}))
	print("X : 2.7, Y : ", sess.run(hypothesis, feed_dict={X:2.7}))
