import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate = .01
batch_size = 100
model_path = "/tmp/model.ckpt"

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])


def multilayer_perceptron(x, weight, bias):
    layer_1 = tf.add(tf.matmul(x, weight['h1']), bias['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weight['h2']), bias['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.add(tf.matmul(layer_2, weight['out']), bias['out'])
    return out_layer


weights = {
    'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random.normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random.normal([n_hidden_1])),
    'b2': tf.Variable(tf.random.normal([n_hidden_2])),
    'out': tf.Variable(tf.random.normal([n_classes]))
}

pred = multilayer_perceptron(X, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

print("Starting 1st session...")
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(3):
        avg_cost = 0.
        total_batch = int(data.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = data.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
            avg_cost += c / total_batch
            print("Epoch: {} - Cost: {:.9f}".format(epoch + 1, avg_cost))
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: data.test.images, Y: data.test.labels}))
    save_path = saver.save(sess, model_path)
    print("Model saved in file: {}".format(save_path))

print("Starting 2nd session...")
with tf.Session() as sess:
    sess.run(init)
    load_path = saver.restore(sess, model_path)
    print("Model restored from file: %s" % save_path)

    for epoch in range(7):
        avg_cost = 0.
        total_batch = int(data.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = data.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
            avg_cost += c / total_batch
            print("Epoch: {} - Cost: {:.9f}".format(epoch + 1, avg_cost))

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: data.test.images, Y: data.test.labels}))
