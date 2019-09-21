import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('monthly-milk-production-pounds.csv', index_col='Month')
data.index = pd.to_datetime(data.index)
train_data = data.head(156)
test_data = data.tail(12)
scl = MinMaxScaler()
train_scaled = scl.fit_transform(train_data)
test_scaled = scl.transform(test_data)

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

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './ex_time_series_model')
    train_seed = list(train_scaled[-12:])
    for i in range(12):
        x_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_prediction = sess.run(outputs, feed_dict={x: x_batch})
        train_seed.append(y_prediction[0, -1, 0])

results = scl.inverse_transform(np.array(train_seed[12:]).reshape(12, 1))
test_data['Generated'] = results
test_data.plot()
plt.show()
