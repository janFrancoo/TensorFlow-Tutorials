import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set Eager API
tf.enable_eager_execution()
tfe = tf.contrib.eager

# Parameters
num_steps = 1000
learning_rate = .01

# Training data
x_train = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313,
                    7.997, 5.654, 9.27, 3.1])
y_train = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65,
                    2.904, 2.42, 2.94, 1.3])

# Weights
w = tfe.Variable(np.random.randn())
b = tfe.Variable(np.random.randn())


# Construct a linear model
def linear_regression(inputs):
    return (inputs * w) + b


# Define loss function
def mean_square_fn(model_fn, inputs, labels):
    return tf.reduce_sum(((model_fn(inputs) - labels) ** 2) / (2 * len(x_train)))


# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# Compute gradients
grad = tfe.implicit_gradients(mean_square_fn)

# Start training
for step in range(num_steps):
    optimizer.apply_gradients(grad(linear_regression, x_train, y_train))

    if (step + 1) % 50 == 0:
        print("Epoch: {}, Loss: {}, W: {}, b: {}".format(step + 1, mean_square_fn(linear_regression, x_train, y_train),
                                                         w.numpy(), b.numpy()))

# Display
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, np.array(w * x_train + b), label='Fitted line')
plt.legend()
plt.show()
