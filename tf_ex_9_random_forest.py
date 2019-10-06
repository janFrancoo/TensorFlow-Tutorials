import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops import resources
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensor_forest.python import tensor_forest

data = input_data.read_data_sets("MNIST_data", one_hot=False)

# Parameters
num_steps = 500
batch_size = 1024
num_classes = 10
num_features = 784
num_trees = 10
max_nodes = 1000

# Input and target
x = tf.placeholder(tf.float32, shape=[None, num_features])
y = tf.placeholder(tf.int32, shape=[None])

# Random forest parameters
h_params = tensor_forest.ForestHParams(num_classes=num_classes, num_features=num_features, num_trees=num_trees,
                                       max_nodes=max_nodes).fill()

# Build forest
forest_graph = tensor_forest.RandomForestGraphs(h_params)

# Training graph and loss
train_op = forest_graph.training_graph(x, y)
loss_op = forest_graph.training_loss(x, y)

# Accuracy
infer_op, _, _ = forest_graph.inference_graph(x)
prediction = tf.argmax(infer_op, 1)
correct_prediction = tf.equal(prediction, tf.cast(y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Init
init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

# Start session
sess = tf.train.MonitoredSession()

# Run Init
sess.run(init_vars)

# Training
for i in range(1, num_steps + 1):
    # Prepare data
    batch_x, batch_y = data.train.next_batch(batch_size)
    _, loss = sess.run([train_op, loss_op], feed_dict={x: batch_x, y: batch_y})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={x: batch_x, y: batch_y})
        print("Step {}: Loss {}, Accuracy {}".format(i, loss, acc))

# Test
test_x, test_y = data.test.images, data.test.labels
print("Test accuracy: ", sess.run(accuracy_op, feed_dict={x: test_x, y: test_y}))

# See results
for i in range(3):
    plt.imshow(data.test.images[i].reshape(28, 28))
    predicted_val = sess.run(prediction, feed_dict={x: data.test.images[i].reshape(1, 784)})
    plt.title("Original value: {} - Predicted value: {}".format(data.test.labels[i], predicted_val[0]))
    plt.show()
