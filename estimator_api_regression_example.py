import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))
y_data = (.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=["X"])
y_df = pd.DataFrame(data=y_data, columns=["Y"])
df = pd.concat([x_df, y_df], axis=1)
print(df.head())

x_plot = []
y_plot = []
for i in range(500):
    random = np.random.randint(len(x_data))
    x_plot.append(df["X"].iloc[random])
    y_plot.append(df["Y"].iloc[random])

plt.scatter(x_plot, y_plot)
plt.show()

batch_size = 8
feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,
                                                    random_state=101)

input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size,
                                                num_epochs=None, shuffle=True)
train_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size,
                                                      num_epochs=1000, shuffle=False)
test_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_test}, y_test, batch_size,
                                                     num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_func, steps=1000)
train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
eval_metrics = estimator.evaluate(input_fn=test_input_func, steps=1000)

print(train_metrics)
print(eval_metrics)

brand_new_data = np.linspace(0, 10, 10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': brand_new_data}, shuffle=False)

predictions = []
for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred['predictions'])

plt.scatter(x_data, y_data)
plt.plot(brand_new_data, predictions, "red")
plt.show()
