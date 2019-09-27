import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib.layers import fully_connected

data = make_blobs(n_samples=100, n_features=3, centers=2, random_state=101)
scl = MinMaxScaler()
scaled_data = scl.fit_transform(data[0])

data_x = scaled_data[:, 0]
data_y = scaled_data[:, 1]
data_z = scaled_data[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_x, data_y, data_z, c=data[1])
plt.show()

num_inputs = 3
num_hidden = 2
num_outputs = num_inputs
num_steps = 1000
learning_rate = 0.01

x = tf.placeholder(tf.float32, shape=[None, num_inputs])
hidden = fully_connected(x, num_hidden, activation_fn=None)
outputs = fully_connected(hidden, num_outputs, activation_fn=None)

loss = tf.reduce_mean(tf.square(outputs - x))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_steps):
        sess.run(train, feed_dict={x: scaled_data})
    output_2d = hidden.eval(feed_dict={x: scaled_data})

plt.scatter(output_2d[:, 0], output_2d[:, 1], c=data[1])
plt.show()
