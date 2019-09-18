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

y1, y2, ts_next_batch = ts_data.next_batch(1, num_time_steps, True)
train_inst = np.linspace(5, 5 + ts_data.resolution * (num_time_steps + 1), num_time_steps + 1)

num_inputs = 1
num_neurons = 100
num_outputs = 1

x = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.GRUCell(
    num_units=num_neurons, activation=tf.nn.relu), output_size=num_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model_2")
    zero_seq_seed = [0.0 for i in range(num_time_steps)]
    for i in range(len(ts_data.x_data)-num_time_steps):
        x_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={x: x_batch})
        zero_seq_seed.append(y_pred[0, -1, 0])

plt.plot(ts_data.x_data, zero_seq_seed, 'b-')
plt.plot(ts_data.x_data[:num_time_steps], zero_seq_seed[:num_time_steps], 'r', linewidth=3)
plt.xlabel('Time')
plt.show()

with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model_2")
    training_instance = np.array([ts_data.y_data[:num_time_steps]])
    for i in range(len(training_instance)-num_time_steps):
        x_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={x: x_batch})
        training_instance.append(y_pred[0, -1, 0])

plt.plot(ts_data.x_data, ts_data.y_data, 'b-')
plt.plot(ts_data.x_data[:num_time_steps], training_instance.flatten()[:num_time_steps], 'r', linewidth=3)
plt.xlabel('Time')
plt.show()
