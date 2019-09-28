import tensorflow as tf

# Basic constant operations
a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph
with tf.Session() as sess:
    print("a: {}, b: {}".format(sess.run(a), sess.run(b)))
    print("Addition with constants: {}".format(sess.run(a+b)))
    print("Multiplication with constants: {}\n".format(sess.run(a*b)))

# Basic operations with variables as graph inputs
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Define operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the default graph
with tf.Session() as sess:
    print("Addition with variables: {}".format(sess.run(add, feed_dict={a: 2, b: 3})))
    print("Multiplication with variables: {}\n".format(sess.run(mul, feed_dict={a: 2, b: 3})))

# Create 2 constants for simple matrix multiplication
# A 1x2 matrix and 2x1 matrix
matrix_1 = tf.constant([[3, 3]])
matrix_2 = tf.constant([[2], [2]])

# Define matrix multiplication operation
product = tf.matmul(matrix_1, matrix_2)

# Launch the default graph
with tf.Session() as sess:
    print("Result of the matrix multiplication: {}".format(sess.run(product)))
