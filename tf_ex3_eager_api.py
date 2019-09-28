import tensorflow as tf

# Set Eager API
tf.enable_eager_execution()
tfe = tf.contrib.eager

# Define constant tensors
a = tf.constant(2)
b = tf.constant(3)

# Run operations without tf.Session()
print("a: {}, b: {}".format(a, b))
c = a + b
d = a * b
print("Addition: a + b = {}".format(c))
print("Multiplication: a * b = {}\n".format(d))

# Create 2 constant tensors
a = tf.constant([[2., 1.],
                 [1., 0.]], dtype=tf.float32)
b = tf.constant([[3., 0.],
                 [5., 1.]], dtype=tf.float32)

# Run operations without tf.Session()
print("a:\n{}\nb:\n{}\n".format(a, b))
c = a + b
d = tf.matmul(a, b)
print("Addition: a + b = \n{}\n".format(c))
print("Multiplication: a * b = \n{}\n".format(d))

# Iterate through tensor a
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])
