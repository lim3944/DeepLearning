#2015004957 임현택 
import tensorflow as tf

sample = "if you want you"
idx2char = list(set(sample))
char2idx = {c: i for i,c in enumerate(idx2char)}

dic_size = len(char2idx)
hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
sequence_length = len(sample)-1
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

X = tf.placeholder(tf.float32, [None, sequence_length])
Y = tf.placeholder(tf.float32, [None])

cell = tf.contrib.rnn.Basic_RNNCell(num_units = hidden_size)
outputs, _state = tf.nn.dynamic_rnn(cell,x_data, dtype=tf.float32)

X_for_fc = tf.reshape(outputs, [-1], hidden_size)
outputs = tf.contrib.layer.fully_connected(X_for_fc, num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size], sequence_length, num_classes)
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs, targest = Y, weights = weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
prediction = tf.argmax(outputs, axis=2)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for epoch in range(50):
	total_cost = 0
	_, cost_val = sess.run([optimizer, cost], feed_dict = {X: x_data, Y: y_data})
	total_cost += cost_val

	print("Epoch : ", '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(total_cost/total_batch))

