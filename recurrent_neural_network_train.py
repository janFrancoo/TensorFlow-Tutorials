import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TimeSeriesData:
    def __init__(self, x_min, x_max, num_points):
        self.x_min = x_min
        self.x_max = x_max
        self.num_points = num_points
        self.resolution = (x_max - x_min) / num_points
        self.x_data = np.linspace(x_min, x_max, num_points)
        self.y_data = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False):
        rand_start = np.random.rand(batch_size, 1)
        ts_start = rand_start * (self.x_max - self.x_min - (steps * self.resolution))
        batch_ts = ts_start + np.arange(0.0, steps + 1) * self.resolution
        y_batch = np.sin(batch_ts)

        if return_batch_ts:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1), batch_ts
        else:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


num_time_steps = 30
ts_data = TimeSeriesData(0, 10, 250)
plt.plot(ts_data.x_data, ts_data.y_data)
plt.show()

y1, y2, ts_next_batch = ts_data.next_batch(1, num_time_steps, True)
plt.plot(ts_next_batch.flatten()[1:], y2.flatten(), '*')
plt.show()

plt.plot(ts_data.x_data, ts_data.y_data, label='sin(t)')
plt.plot(ts_next_batch.flatten()[1:], y2.flatten(), '*', label='Single training instance')
plt.legend()
plt.tight_layout()
plt.show()

train_inst = np.linspace(5, 5 + ts_data.resolution * (num_time_steps + 1), num_time_steps + 1)
plt.title('A Training Instance')
plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), 'bo', markersize=15, alpha=0.5, label='Instance')
plt.plot(train_inst[1:], ts_data.ret_true(train_inst[1:]), 'ko', markersize=7, label='Target')
plt.legend()
plt.show()

num_inputs = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.0001
num_train_iterations = 2000
batch_size = 1

x = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(
    num_units=num_neurons, activation=tf.nn.relu), output_size=num_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_train_iterations):
        x_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
        sess.run(train, feed_dict={x: x_batch, y: y_batch})
        if i % 100 == 0:
            mse = loss.eval(feed_dict={x: x_batch, y: y_batch})
            print(i, "\tMSE:", mse)
    saver.save(sess, "./rnn_time_series_model")
