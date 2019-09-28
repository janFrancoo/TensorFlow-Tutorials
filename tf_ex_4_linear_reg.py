import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Parameters
epochs = 1000
learning_rate = .01

# Training data
x_train = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313,
                    7.997, 5.654, 9.27, 3.1])
y_train = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65,
                    2.904, 2.42, 2.94, 1.3])

# Inputs for graph
x = tf.placeholder(tf.float64)
y = tf.placeholder(tf.float64)

# Weights
w = tf.Variable(np.random.randn(), dtype=tf.float64)
b = tf.Variable(np.random.randn(), dtype=tf.float64)

# Construct a linear model
predict = tf.add(tf.multiply(x, w), b)

# Set loss function and optimizer
loss = tf.reduce_sum(((predict - y) ** 2)/(2*x_train.shape[0]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for (x_values, y_values) in zip(x_train, y_train):
            sess.run(optimizer, feed_dict={x: x_values, y: y_values})

        if (epoch+1) % 50 == 0:
            c = sess.run(loss, feed_dict={x: x_train, y: y_train})
            print("Epoch: {}, Loss: {}, W: {}, b: {}".format(epoch + 1, c, sess.run(w), sess.run(b)))

    # Display
    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, (sess.run(w) * x_train) + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
