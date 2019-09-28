import tensorflow as tf

# Create an operation called hello
hello = tf.constant('Hello TensorFlow!')

# Start a session
sess = tf.Session()

# Run graph
print(sess.run(hello))
