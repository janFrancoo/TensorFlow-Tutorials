import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Import data
data = input_data.read_data_sets('/MNIST_data', one_hot=True)

# Split data to train and test
x_train, y_train = data.train.next_batch(5000)
x_test, y_test = data.test.next_batch(200)

# Inputs for graph
x_tr = tf.placeholder(tf.float32, [None, 784])
x_ts = tf.placeholder(tf.float32, [784])

# NN calculation using L2 distance
distance = tf.reduce_sum(tf.abs(tf.add(x_tr, tf.negative(x_ts))), reduction_indices=1)

# Prediction (get minimum distance)
prediction = tf.argmin(distance, 0)

# Define acc, init func and create a list for visualizing the results
preds = []
accuracy = 0.
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(x_test)):
        # Get nearest neighbour:
        nn_index = sess.run(prediction, feed_dict={x_tr: x_train, x_ts: x_test[i, :]})
        # Get nearest neighbour class label and compare
        print("Test: {} Prediction: {} True class: {}".format(i, np.argmax(y_train[nn_index]), np.argmax(y_test[i])))
        preds.append((nn_index, np.argmax(y_test[i])))
        if np.argmax(y_train[nn_index]) == np.argmax(y_test[i]):
            accuracy += 1. / len(x_test)
        else:
            plt.imshow(x_train[nn_index].reshape(28, 28))
            plt.title(np.argmax(y_test[i]))
            plt.show()
    print("Accuracy:", accuracy)

# Visualize
i = 0
limit = 5
for index, predicted_label in preds:
    plt.imshow(x_train[index].reshape(28, 28))
    plt.title(predicted_label)
    plt.show()
    i += 1
    if i == limit:
        break
