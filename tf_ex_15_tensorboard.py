import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import data
data = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
epochs = 25
batch_size = 100
learning_rate = .01
logs_path = "/tmp/TensorFlow_logs/ex/"

# Graph inputs
X = tf.placeholder(tf.float32, [None, 784], name='InputData')
Y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

# Weights
W = tf.Variable(tf.zeros([784, 10]), name='Weights')
b = tf.Variable(tf.zeros([10]), name='Bias')

# Construct model with names
with tf.name_scope('Model'):
    pred = tf.nn.softmax(tf.matmul(X, W) + b)

with tf.name_scope('Loss'):
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=1))

with tf.name_scope('SGD'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1)), tf.float32))

# Init func
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Start Training
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(data.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = data.train.next_batch(batch_size)
            _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={X: batch_xs, Y: batch_ys})
            summary_writer.add_summary(summary, epoch * total_batch + i)
            avg_cost += c / total_batch
        print("Epoch: {} -- Cost = {:.9f}".format(epoch + 1, avg_cost))
    print("Accuracy: {}".format(acc.eval({X: data.test.images, Y: data.test.labels})))
    print("Run the command line:\n--> tensorboard --logdir=/tmp/tensorflow_logs\n"
          "Then open http://0.0.0.0:6006/ into your web browser")
