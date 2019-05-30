import tensorflow as tf

constant_value = tf.constant("Deep Learning course")
print(constant_value)

ten = tf.constant(10)
nine = tf.constant(9)
nineteen = tf.add(ten,nine)
print(nineteen)

constant_array = tf.constant([1,2])
print(constant_array)

print("==============")

sess = tf.Session()
print(sess.run(constant_value))
print(sess.run([ten,nine,nineteen]))
print(sess.run(constant_array))

sess.close()