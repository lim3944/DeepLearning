import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

sentence = ("if you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.")
char_set = list(set(sentence))
char_dic = {w: i for i,w in enumerate(char_set)}

dic_size = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10
learning_rate = 0.1

x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    x_data.append(x)
    y_data.append(y)

batch_size = len(x_data)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

x_one_hot = tf.one_hot(X,num_classes)
cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size, state_is_tuple = True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, states = tf.nn.dynamic_rnn(cell,x_one_hot, initial_state=initial_state, dtype = tf.float32)

X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn = None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs, targets = Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis = 2)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(500):
		l, _ = sess.run([loss,train], feed_dict = {X: x_data, Y : y_data})
		'''
		for j, result in enumerate(results):
			index = np.argmax(result, axis=1)
			print(i,j,''.join([char_set[t] for t in index]), l)
		'''
		#result = sess.run(prediction, feed_dict={X:x_data})
		#result_str = [char_set[c] for c in np.squeeze(result)]
		#print(i, "loss:", l, "Prediction:", ''.join(result_str))
		print(i, "loss",l)

	results = sess.run(outputs,feed_dict={X:x_data})

	for j, result in enumerate(results):
		index =	np.argmax(result,axis=1)
		if j is 0:
			print(''.join([char_set[t] for t in index]), end='')
		else :
			print(char_set[index[-1]], end='')