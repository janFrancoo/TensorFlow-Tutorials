import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
num_steps = 500
batch_size = 128
learning_rate = 0.001

drop_out = .75
num_input = 784
num_classes = 10

# Graph inputs
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)


# Convolution process
def conv_2d(x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# Max pooling
def max_pool_2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, weights_v, biases_v, drop_out_v):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv_1 = conv_2d(x, weights_v['wc1'], biases_v['bc1'])
    conv_1 = max_pool_2d(conv_1, k=2)

    conv_2 = conv_2d(conv_1, weights_v['wc2'], biases_v['bc2'])
    conv_2 = max_pool_2d(conv_2, k=2)

    fc_1 = tf.reshape(conv_2, [-1, weights_v['wd1'].get_shape().as_list()[0]])
    fc_1 = tf.add(tf.matmul(fc_1, weights_v['wd1']), biases_v['bd1'])
    fc_1 = tf.nn.relu(fc_1)
    fc_1 = tf.nn.dropout(fc_1, drop_out_v)

    out = tf.add(tf.matmul(fc_1, weights_v['out']), biases_v['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Init func
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    for step in range(1, num_steps+1):
        batch_x, batch_y = data.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: drop_out})
        if step % 100 == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
            print("Step {}: Loss = {:.4f} - Accuracy = {:.3f}".format(step, loss, acc))

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: data.test.images[:256], Y: data.test.labels[:256],
                                                             keep_prob: 1.0}))
