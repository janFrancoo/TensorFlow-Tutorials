import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Read data
data = input_data.read_data_sets("MNIST_data", one_hot=False)

# Parameters
learning_rate = .1
num_steps = 1000
batch_size = 128

num_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
num_classes = 10

# Input func for training
input_fn = tf.estimator.inputs.numpy_input_fn(x={"images": data.train.images}, y=data.train.labels,
                                              batch_size=batch_size, num_epochs=None, shuffle=True)


# Define the neural network
def neural_net(x_dict):
    x = x_dict["images"]
    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    out = tf.layers.dense(layer_2, num_classes)
    return out


# Define the model func
def model_fn(features, labels, mode):
    logits = neural_net(features)
    pred_probas = tf.nn.softmax(logits)
    pred_classes = tf.argmax(pred_probas, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                            labels=tf.cast(labels, dtype=tf.int32)))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).\
        minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    estimator_specs = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes, loss=loss_op, train_op=train_op,
                                                 eval_metric_ops={'accuracy': acc_op})

    return estimator_specs


# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Train the Model
model.train(input_fn, steps=num_steps)

# Define the input func for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': data.test.images}, y=data.test.labels,
                                              batch_size=batch_size, shuffle=False)

# Use the Estimator 'evaluate' method
print(model.evaluate(input_fn))

# Predict single images
n_images = 4
test_images = data.test.images[:n_images]
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': test_images}, shuffle=False)
preds = list(model.predict(input_fn))

for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.title("Model prediction: {}".format(preds[i]))
    plt.show()
