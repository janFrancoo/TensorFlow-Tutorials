import tensorflow as tf
import tensorflow.estimator as estimator
from sklearn.datasets import load_wine
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

feat_cols = [tf.feature_column.numeric_column('x', shape=[13])]

deep_model = estimator.DNNClassifier(hidden_units=[13, 13, 13], feature_columns=feat_cols, n_classes=3,
                                     optimizer=tf.train.GradientDescentOptimizer(learning_rate=.01))

input_func = estimator.inputs.numpy_input_fn(x={'x': scaled_x_train}, y=y_train, batch_size=10, num_epochs=5,
                                             shuffle=False)

input_func_eval = estimator.inputs.numpy_input_fn(x={'x': scaled_x_test}, shuffle=False)

deep_model.train(input_fn=input_func, steps=500)
predictions = list(deep_model.predict(input_fn=input_func_eval))
predictions_list = [p['class_ids'][0] for p in predictions]

print(classification_report(y_test, predictions_list))
print(confusion_matrix(y_test, predictions_list))
