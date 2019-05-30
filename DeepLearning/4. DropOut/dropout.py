#2015004957 임현택 
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([784, 128],-1.0,1.0))
b1 = tf.Variable(tf.random_uniform([128],-1.0,1.0))

W2 = tf.Variable(tf.random_uniform([128,32],-1.0,1.))
b2 = tf.Variable(tf.random_uniform([32],-1.0,1.0))

W3 = tf.Variable(tf.random_uniform([32,10],-1.0,1.0))
b3 = tf.Variable(tf.random_uniform([10],-1.0,1.0))

L1 = tf.sigmoid(tf.matmul(X,W1)+b1)
L1 = tf.nn.dropout(L1, keep_prob)
L2 = tf.sigmoid(tf.matmul(L1,W2)+b2)
model = tf.matmul(L2, W3)+b3


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(50):
	total_cost = 0

	for i in range(total_batch):
		batch_xs,batch_ys = mnist.train.next_batch(batch_size)

		_, cost_val = sess.run([optimizer, cost], feed_dict = {X: batch_xs, Y: batch_ys, keep_prob:0.75})
		total_cost += cost_val

	print("Epoch : ", '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(total_cost/total_batch))

is_correct = tf.equal(tf.arg_max(model, 1), tf.math.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("Accuracy : ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1}))