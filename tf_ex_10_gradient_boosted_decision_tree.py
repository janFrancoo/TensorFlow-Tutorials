import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier

# Silence warnings and display logs
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(tf.logging.INFO)

# Import data
data = input_data.read_data_sets("MNIST_data", one_hot=False)

# Parameters
batch_size = 4096
num_classes = 10
num_features = 784
max_steps = 10000

# GBDT parameters
learning_rate = 0.1
l1_regularization = 0.
l2_regularization = 1.
examples_per_layer = 1000
num_trees = 10
max_depth = 16

# Configuration
learner_config = gbdt_learner.LearnerConfig()
learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
learner_config.regularization.l1 = l1_regularization
learner_config.regularization.l2 = l2_regularization / examples_per_layer
learner_config.constraints.max_tree_depth = max_depth
learner_config.growing_mode = gbdt_learner.LearnerConfig.LAYER_BY_LAYER
run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)
learner_config.multi_class_strategy = gbdt_learner.LearnerConfig.DIAGONAL_HESSIAN

# TensorFlow GBDT estimator
gbdt_model = GradientBoostedDecisionTreeClassifier(model_dir=None, learner_config=learner_config,
                                                   n_classes=num_classes, examples_per_layer=examples_per_layer,
                                                   num_trees=num_trees, center_bias=False, config=run_config)

# Define input func for training
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': data.train.images}, y=data.train.labels,
                                              batch_size=batch_size, num_epochs=None, shuffle=True)

# Train
gbdt_model.fit(input_fn=input_fn, max_steps=max_steps)

# Define input func for testing
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': data.test.images}, y=data.test.labels,
                                              batch_size=batch_size, shuffle=False)

# Test
e = gbdt_model.evaluate(input_fn=input_fn)
print("Testing Accuracy:", e['accuracy'])
