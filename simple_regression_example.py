import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
plt.scatter(x, y)
plt.show()

m = tf.Variable(0.44)
b = tf.Variable(0.87)
error = 0

for x_data, y_data in zip(x, y):
    y_calc = (m * x_data) + b
    error += (y_data - y_calc) ** 2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    step = 100

    sess.run(init)
    sess.run(train)
    m_calc, b_calc = sess.run([m, b])

    res = (m_calc * x) + b_calc
    plt.plot(x, res, 'r')
    plt.scatter(x, y)
    plt.show()

    for i in range(step):
        sess.run(train)

    m_calc, b_calc = sess.run([m, b])

res = (m_calc * x) + b_calc
plt.plot(x, res, 'r')
plt.scatter(x, y)
plt.show()
