import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))
y_data = (.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=["X"])
y_df = pd.DataFrame(data=y_data, columns=["Y"])
df = pd.concat([x_df, y_df], axis=1)
print(df.head())

plt.scatter(x_df.sample(n=750, random_state=101), y_df.sample(n=750, random_state=101))
plt.show()

batch_size = 8
m = tf.Variable(0.81)
b = tf.Variable(0.17)
x_ph = tf.placeholder(tf.float32, [batch_size])
y_ph = tf.placeholder(tf.float32, [batch_size])

y_true = (m * x_ph) + b
error = tf.reduce_sum(tf.square(y_ph - y_true))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    batches = 1000
    sess.run(init)

    for i in range(batches):
        random_index = np.random.randint(len(x_data), size=batch_size)
        sess.run(train, feed_dict={x_ph: df["X"].iloc[random_index], y_ph: df["Y"].iloc[random_index]})

    true_m, true_b = sess.run([m, b])

y_true_plot = (true_m * x_data) + true_b

plt.scatter(x_df, y_df)
plt.plot(x_data, y_true_plot, "red")
plt.show()
