import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Import data
data = input_data.read_data_sets("/MNIST_data", one_hot=True)

# Parameters
epochs = 25
learning_rate = .01
batch_size = 100

# Inputs for graph
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Set weights
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
predict = tf.nn.softmax(tf.matmul(x, w) + b)

# Define loss function and optimizer, initialize variables
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(predict), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(25):
        avg_lost = 0
        total_batch = data.train.num_examples // batch_size
        for i in range(total_batch):
            batch_xs, batch_ys = data.train.next_batch(batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys})
            avg_lost += c / total_batch
        print("Epoch: {}, Loss: {}".format(epoch+1, avg_lost))
    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: data.test.images[:3000], y: data.test.labels[:3000]}))
    predict_img = tf.argmax(predict, 1)
    predicted_values = predict_img.eval({x: data.test.images[:10]})
    for i in range(10):
        plt.title(predicted_values[i])
        plt.imshow(data.test.images[i].reshape(28, 28))
        plt.show()
