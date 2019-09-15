import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_values = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_values)


def convolution_2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolution_layer(input_x, shape):
    w = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(convolution_2d(input_x, w) + b)


def fully_connected_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    w = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, w) + b


data = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
hold_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])
convolution_1 = convolution_layer(x_image, shape=[5, 5, 1, 32])
pooling_1 = max_pool_2x2(convolution_1)
convolution_2 = convolution_layer(pooling_1, shape=[5, 5, 32, 64])
pooling_2 = max_pool_2x2(convolution_2)
flat = tf.reshape(pooling_2, [-1, 7*7*64])
fc_1 = tf.nn.relu(fully_connected_layer(flat, 1024))
drop_out = tf.nn.dropout(fc_1, keep_prob=hold_prob)
y_prediction = fully_connected_layer(drop_out, 10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_prediction))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
steps = 5000

with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        batch_x, batch_y = data.train.next_batch(50)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
        if i % 100 == 0:
            print("ON STEP: {}".format(i))
            matches = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            print("ACCURACY: ", sess.run(acc, feed_dict={x: data.test.images, y_true: data.test.labels, hold_prob: 1.0}))
