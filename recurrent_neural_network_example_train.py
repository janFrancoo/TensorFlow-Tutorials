import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('monthly-milk-production-pounds.csv', index_col='Month')
print(data.head())
data.plot()
plt.show()

data.index = pd.to_datetime(data.index)
train_data = data.head(156)
test_data = data.tail(12)
scl = MinMaxScaler()
train_scaled = scl.fit_transform(train_data)
test_scaled = scl.transform(test_data)
print(train_scaled)
print(test_scaled)

def next_batch(training_data, steps):
    random_start = np.random.randint(0, len(training_data)-steps)
    y_data = np.array(training_data[random_start:random_start+steps+1]).reshape(1, steps+1)
    return y_data[:, :-1].reshape(-1, steps, 1), y_data[:, 1:].reshape(-1, steps, 1)

num_inputs = 1
num_outputs = 1
num_neurons = 100
num_time_steps = 12
learning_rate = 0.001
num_training_iterations = 4000

x = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.relu), output_size=num_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for i in range(num_training_iterations):
        x_batch, y_batch = next_batch(train_scaled, num_time_steps)
        sess.run(train, feed_dict={x: x_batch, y: y_batch})
        if i % 100 == 0:
            mse = loss.eval(feed_dict={x: x_batch, y: y_batch})
            print(i, "\tMSE: ", mse)

    saver.save(sess, './ex_time_series_model')
