import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Read data
data = input_data.read_data_sets("/MNIST_data", one_hot=True)

# Parameters
num_steps = 500
batch_size = 128
display_step = 100
learning_rate = 0.1

n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784
num_classes = 10

# Graph input
x = tf.placeholder(tf.float32, [None, num_input])
y = tf.placeholder(tf.float32, [None, num_classes])

# Store weight and bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_network(xx):
    layer_1 = tf.add(tf.matmul(xx, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
logits = neural_network(x)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

# Evaluate model
prediction = tf.argmax(logits, 1)
c_prediction = tf.equal(prediction, tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(c_prediction, tf.float32))

# Init func
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    for step in range(1, num_steps+1):
        batch_x, batch_y = data.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, accuracy = sess.run([loss_op, acc], feed_dict={x: batch_x, y: batch_y})
            print("Step: {}, Batch Loss: {:.4f}, Accuracy: {:.3f}".format(step, loss, accuracy))
    print("Optimization finished!")
    acc = sess.run(acc, feed_dict={x: data.test.images, y: data.test.labels})
    predictions = sess.run(prediction, feed_dict={x: data.test.images[:5], y: data.test.labels[:5]})
    print("Test Accuracy: {}".format(acc))
    for i in range(5):
        plt.imshow(data.test.images[i].reshape(28, 28))
        plt.title(predictions[i])
        plt.show()
