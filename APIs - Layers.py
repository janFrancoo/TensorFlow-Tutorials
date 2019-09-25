import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_wine
import tensorflow.contrib.layers as layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = load_wine()
feat_data = data['data']
labels = data['target']

x_train, x_test, y_train, y_test = train_test_split(feat_data, labels, test_size=.3, random_state=101)

scl = MinMaxScaler()
scaled_x_train = scl.fit_transform(x_train)
scaled_x_test = scl.fit_transform(x_test)

one_hot_y_train = pd.get_dummies(y_train).as_matrix()
one_hot_y_test = pd.get_dummies(y_test).as_matrix()

num_feat = 13
num_hidden1 = 13
num_hidden2 = 13
num_outputs = 3
training_steps = 1000

x = tf.placeholder(tf.float32, shape=[None, num_feat])
y_true = tf.placeholder(tf.float32, shape=[None, num_outputs])

act_func = tf.nn.relu
hidden1 = layers.fully_connected(x, num_hidden1, activation_fn=act_func)
hidden2 = layers.fully_connected(hidden1, num_hidden2, activation_fn=act_func)
output = layers.fully_connected(hidden2, num_outputs)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=output)
optimizer = tf.train.AdamOptimizer(learning_rate=.01)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(training_steps):
        sess.run(train, feed_dict={x: scaled_x_train, y_true: one_hot_y_train})
    logits = output.eval(feed_dict={x: scaled_x_test})
    preds = tf.argmax(logits, axis=1)
    results = preds.eval()

print(classification_report(y_test, results))
print(confusion_matrix(y_test, results))
